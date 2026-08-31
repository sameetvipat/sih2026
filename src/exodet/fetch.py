"""Shared, resumable, rate-limit-aware download harness.

Both the real-label fetcher and the quiet-baseline bank builder pull light
curves from MAST, which is network-bound and rate-limited.  Rather than
maintain two divergent download harnesses, they share this one.

Design points that are not obvious:

* **Exponential backoff with jitter.**  MAST responds to rate limiting far
  better than to fixed-interval retries; jitter stops a pool of workers from
  synchronising into a thundering herd after a shared failure.
* **stdout is redirected once, in the caller's thread, around the whole pool.**
  lightkurve prints a progress bar per file straight to stdout.  Doing the
  redirect per worker with contextlib.redirect_stdout does not work: it mutates
  the global sys.stdout and is explicitly not thread-safe, so concurrent
  enter/exit pairs restore each other's already-closed buffers and unrelated
  targets die with "I/O operation on closed file".
* **Every network call carries an enforced wall-clock deadline.**  `with_backoff`
  retries *raised* exceptions, but a stalled socket -- connection accepted, no
  bytes ever delivered -- raises nothing and blocks forever, so backoff never
  sees it.  That is not hypothetical: it stalled a full run at 95% completion.
  Two independent layers now cover it, because neither is sufficient alone.
  `socket.setdefaulttimeout` catches a stalled *read* inside library code that
  exposes no timeout parameter (astroquery/lightkurve do not), and
  `call_with_deadline` caps *total* wall-clock time, which a per-read timeout
  cannot do -- a server dribbling one byte per second resets the socket timeout
  forever while making no real progress.  Both surface as ordinary exceptions,
  so they feed the existing backoff path rather than a second, divergent one.
* **Shards are written atomically** (temp file then os.replace) so an interrupt
  can never leave a half-written parquet, and a corrupt shard is skipped rather
  than being fatal on resume.
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import random
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# MAST tolerates a moderate pool; pushing higher trades throughput for
# throttling and connection rejections.
DEFAULT_JOBS = 12
MAX_JOBS = 20


# Per-read socket stall ceiling.  This is NOT a total-download budget: it
# bounds how long a socket may sit with no bytes arriving.  Set it well above
# MAST's worst honest latency, or healthy-but-slow transfers get killed and the
# "fix" looks like a network outage.
SOCKET_STALL_SECONDS = 90.0

# Total wall-clock ceiling for one target's search+download.  A whole Kepler
# quarter over a congested link can legitimately take minutes, so this is
# deliberately generous -- it exists to break infinite hangs, not to enforce
# throughput.
DOWNLOAD_DEADLINE_SECONDS = 420.0


def install_socket_timeout(seconds: float = SOCKET_STALL_SECONDS) -> None:
    """Apply a process-wide default socket timeout.

    astroquery and lightkurve build their own HTTP sessions and expose no
    timeout parameter to pass down, so this global is the only place a read
    stall inside them can be bounded.  It applies to sockets created after this
    call, which is why the harness installs it at import time.
    """
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(seconds)


install_socket_timeout()


class DownloadTimeout(TimeoutError):
    """A network call exceeded its wall-clock deadline.

    Subclasses TimeoutError (and so Exception) deliberately: `with_backoff`
    then retries a hang through exactly the same path it retries a refused
    connection, instead of needing a parallel timeout-specific retry branch.
    """


def call_with_deadline(fn: Callable, *args, timeout: float, **kwargs):
    """Run `fn` on a daemon thread and abandon it if it outlives `timeout`.

    A daemon thread rather than ThreadPoolExecutor.submit().result(timeout=...):
    neither can actually kill a thread blocked in a C-level socket read, but a
    pool's workers are non-daemon and joined at interpreter exit, so an
    abandoned hung worker would deadlock process shutdown -- turning a stalled
    download into a stalled *exit*.  A daemon thread is abandoned cleanly and
    dies with the process.

    The abandoned thread does keep its connection until the socket timeout above
    trips it, so the two layers are complementary rather than redundant.
    """
    box: dict = {}

    def runner():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:             # noqa: BLE001 - re-raised below
            box["error"] = exc

    th = threading.Thread(target=runner, daemon=True,
                          name=f"deadline-{getattr(fn, '__name__', 'call')}")
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise DownloadTimeout(
            f"{getattr(fn, '__name__', 'call')} exceeded {timeout:.0f}s "
            f"wall-clock deadline")
    if "error" in box:
        raise box["error"]
    return box.get("value")


# --------------------------------------------------------------------------- #
# retry
# --------------------------------------------------------------------------- #
def with_backoff(fn: Callable, *args, attempts: int = 4, base: float = 2.0,
                 cap: float = 30.0, rng: random.Random | None = None, **kwargs):
    """Call `fn`, retrying transient failures with exponential backoff + jitter.

    Raises the last exception if every attempt fails, so the caller can log the
    target and move on rather than losing the whole batch.
    """
    rng = rng or random.Random()
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                 # noqa: BLE001 - caller logs it
            last = exc
            if i == attempts - 1:
                break
            delay = min(base ** i, cap) * (0.5 + rng.random())   # full jitter
            time.sleep(delay)
    raise last


# A genuine Kepler/TESS light curve FITS is >100 KB. Anything at or below the
# 64 KB buffer boundary is a partial transfer that MAST dropped mid-flight.
TRUNCATED_FITS_BYTES = 80 * 1024


def _target_digits(target: str) -> str:
    """Numeric portion of a KIC/TIC identifier, for cache path matching."""
    return "".join(ch for ch in target if ch.isdigit())


def purge_truncated_cache(target: str | None = None) -> int:
    """Delete partially-downloaded FITS files from lightkurve's cache.

    This is what makes retrying worthwhile.  When MAST drops a transfer,
    lightkurve still caches the truncated file, and every subsequent attempt
    reads that cached stub instantly instead of re-downloading -- so backoff
    alone retries forever against a corrupt local copy.  Measured under 18
    concurrent downloads: 69 of 391 cached files were truncated to exactly
    64 KB, failing 36% of targets.  Removing the stub lets the retry actually
    re-fetch.
    """
    root = os.path.expanduser("~/.cache/lightkurve")
    if not os.path.isdir(root):
        return 0
    pattern = os.path.join(root, "**", "*.fits")
    digits = _target_digits(target) if target else None
    removed = 0
    for path in glob.iglob(pattern, recursive=True):
        try:
            if os.path.getsize(path) > TRUNCATED_FITS_BYTES:
                continue
            if digits and digits.lstrip("0") not in path.replace("0", "0"):
                # match on the un-padded id appearing anywhere in the path
                if digits.lstrip("0") not in "".join(
                        ch for ch in path if ch.isdigit()):
                    continue
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


# --------------------------------------------------------------------------- #
# light curve download
# --------------------------------------------------------------------------- #
def download_lightcurve(target: str, mission: str, max_segments: int = 1,
                        max_days: float | None = 45.0,
                        segment: int | None = None,
                        deadline: float = DOWNLOAD_DEADLINE_SECONDS):
    """Fetch a light curve as raw (time, flux, flux_err) arrays.

    Returns None when nothing usable is available.  Flux is NOT normalised or
    detrended -- callers must feed the raw arrays through `prepare()` so that
    training data and live queries pass through identical preprocessing.

    Kepler Quarter 0 is skipped: it is a ~10-day commissioning run (469 points
    on one test target), far too short to search for periods up to 13 days.
    Quarter 1 is ~33 days, conveniently close to a TESS sector.

    `segment` picks which observing window to take (0 = first available after
    the Q0 exclusion).  Baseline banks should vary it: instrumental systematics
    are epoch-specific -- momentum dumps, thermal settling and pointing drift
    differ quarter to quarter -- so a bank drawn entirely from one quarter would
    encode that quarter's quirks as if they were the universal noise floor.
    """
    import lightkurve as lk

    # Both the catalogue query and the file transfer are bounded: the query is
    # the call that stalled the 95%-complete run, and it is the cheaper of the
    # two, so it gets the tighter share of the budget.
    search_deadline = min(120.0, deadline * 0.3)
    if mission == "kepler":
        q = call_with_deadline(lk.search_lightcurve, target, mission="Kepler",
                               author="Kepler", cadence="long",
                               timeout=search_deadline)
        if q is None or len(q) == 0:
            return None
        missions = [str(m) for m in q.table["mission"]]
        keep = [i for i, m in enumerate(missions) if "Quarter 00" not in m]
        if not keep:
            return None
        q = q[keep]
    else:
        q = call_with_deadline(lk.search_lightcurve, target, mission="TESS",
                               author="SPOC", exptime=120,
                               timeout=search_deadline)
    if q is None or len(q) == 0:
        return None

    if segment is not None and len(q) > 1:
        i = segment % len(q)
        q = q[i:i + max_segments]
    else:
        q = q[:max_segments]

    try:
        coll = call_with_deadline(q.download_all,
                                  timeout=max(deadline - search_deadline, 60.0))
        if coll is None or len(coll) == 0:
            return None
        lc = coll.stitch().remove_nans()
    except Exception:
        # Almost always a truncated cached file. Drop the stub so the caller's
        # retry re-fetches instead of re-reading the same corrupt bytes.  A
        # timed-out transfer leaves exactly the same kind of stub, so the purge
        # matters at least as much on the deadline path as on the error path.
        purge_truncated_cache(target)
        raise

    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)
    e = np.asarray(lc.flux_err.value, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f, e = t[good], f[good], e[good]

    if max_days and t.size and (t.max() - t.min()) > max_days:
        keep = t <= (t.min() + max_days)
        t, f, e = t[keep], f[keep], e[keep]

    if t.size < 500:
        return None
    return t, f, (e if np.isfinite(e).all() else None)


# --------------------------------------------------------------------------- #
# shard checkpointing
# --------------------------------------------------------------------------- #
def load_shards(shard_dir: str) -> pd.DataFrame:
    """Concatenate completed shards, skipping any left corrupt by a kill."""
    frames = []
    for path in sorted(glob.glob(os.path.join(shard_dir, "part_*.parquet"))):
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:                 # truncated mid-write
            print(f"[warn] unreadable shard {path}: {exc}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flush_shard(buf: list[dict], shard_dir: str, index: int) -> int:
    """Atomically write `buf` to a new shard and clear it. Returns next index."""
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(shard_dir, f"part_{index:04d}.parquet")
    tmp = path + ".tmp"
    pd.DataFrame(buf).to_parquet(tmp, index=False)
    os.replace(tmp, path)                        # atomic: no half-written files
    buf.clear()
    return index + 1


def next_shard_index(shard_dir: str) -> int:
    return len(glob.glob(os.path.join(shard_dir, "part_*.parquet")))


# --------------------------------------------------------------------------- #
# the pool
# --------------------------------------------------------------------------- #
def run_resumable_pool(tasks: Sequence, work_fn: Callable, shard_dir: str,
                       jobs: int = DEFAULT_JOBS, flush_every: int = 25,
                       flush_seconds: float = 60.0,
                       desc: str = "fetching",
                       failures_path: str | None = None,
                       stall_seconds: float = 900.0) -> pd.DataFrame:
    """Run `work_fn` over `tasks` on a bounded thread pool, checkpointing.

    `work_fn(task) -> dict | None`.  A returned dict carrying a non-null
    "error" key is treated as a failure: it is still recorded (so the run is
    auditable) but callers should retry it on resume rather than caching it as
    a defeat.  A single failing target must never halt the batch.

    Checkpointing is triggered by whichever of `flush_every` rows or
    `flush_seconds` elapsed comes first.  The count alone is not enough for
    network-bound work: at ~20 s per download, a 25-row threshold means a kill
    can discard eight minutes of completed downloads.  A deliberate
    kill-and-resume test with count-only flushing lost every row.
    """
    from tqdm import tqdm

    jobs = max(1, min(jobs, MAX_JOBS))
    buf: list[dict] = []
    shard_i = next_shard_index(shard_dir)
    failures: list[dict] = []
    last_flush = time.monotonic()

    # Batch-level watchdog: the per-call deadlines above should make this
    # unreachable, but "should" is what the last run assumed too.  It watches
    # completed *results*, not sockets, so it catches a stall anywhere in the
    # pool -- including a hang in feature extraction rather than the download.
    progress = {"at": time.monotonic(), "n": 0}
    stop_watchdog = threading.Event()

    def watchdog():
        while not stop_watchdog.wait(30.0):
            idle = time.monotonic() - progress["at"]
            if idle > stall_seconds:
                print(f"\n[stall] no completed target in {idle/60:.1f} min "
                      f"({progress['n']}/{len(tasks)} done). Per-call deadlines "
                      f"should have prevented this -- the pool is wedged "
                      f"somewhere they do not cover.", file=sys.stderr)
                progress["at"] = time.monotonic()   # warn again, don't spam

    wd = threading.Thread(target=watchdog, daemon=True, name="fetch-watchdog")
    wd.start()

    try:
        # One redirect, in this thread only, for the lifetime of the pool.
        # tqdm writes to stderr, so progress still shows.
        with contextlib.redirect_stdout(io.StringIO()):
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(work_fn, t): t for t in tasks}
                for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                    try:
                        row = fut.result()
                    except Exception as exc:     # work_fn itself blew up
                        row = {"target": str(futs[fut]),
                               "error": f"{type(exc).__name__}: {exc}"[:200]}
                    if row is None:
                        continue
                    if row.get("error"):
                        failures.append(row)
                    buf.append(row)
                    progress["at"] = time.monotonic()
                    progress["n"] += 1
                    due = (len(buf) >= flush_every
                           or (time.monotonic() - last_flush) >= flush_seconds)
                    if due:
                        shard_i = flush_shard(buf, shard_dir, shard_i)
                        last_flush = time.monotonic()
    except KeyboardInterrupt:
        print("\ninterrupted -- flushing completed rows", file=sys.stderr)
    finally:
        stop_watchdog.set()
        if buf:
            flush_shard(buf, shard_dir, shard_i)

    if failures_path and failures:
        os.makedirs(os.path.dirname(failures_path) or ".", exist_ok=True)
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"[info] {len(failures)} failures logged -> {failures_path}",
              file=sys.stderr)

    return load_shards(shard_dir)


def merge_preferring_success(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Deduplicate on `key`, keeping a successful row over an earlier failure."""
    if df.empty:
        return df
    if "error" not in df:
        return df.drop_duplicates(subset=[key], keep="first")
    return (df.assign(_ok=df["error"].isna())
              .sort_values("_ok", ascending=False)
              .drop_duplicates(subset=[key], keep="first")
              .drop(columns="_ok"))
