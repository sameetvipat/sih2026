"""Tests for resumable dataset generation.

These exercise the shard/resume machinery directly with synthetic rows, so they
run in milliseconds rather than invoking the real pipeline.
"""
import os
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_dataset import finish, flush, load_shards  # noqa: E402


def test_flush_writes_a_shard_and_clears_the_buffer(tmp_path):
    buf = [{"label": "transit", "seed": i, "sde": 10.0} for i in range(3)]
    nxt = flush(buf, str(tmp_path), 0)
    assert nxt == 1
    assert buf == [], "buffer should be cleared after flushing"
    assert (tmp_path / "part_0000.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "part_0000.parquet")) == 3


def test_flush_leaves_no_temp_files(tmp_path):
    """Writes go through a .tmp then os.replace, so a kill cannot half-write."""
    flush([{"label": "noise", "seed": 1}], str(tmp_path), 0)
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_shards_concatenates_in_order(tmp_path):
    flush([{"label": "transit", "seed": 1}], str(tmp_path), 0)
    flush([{"label": "eclipse", "seed": 2}], str(tmp_path), 1)
    df = load_shards(str(tmp_path))
    assert len(df) == 2
    assert set(df["seed"]) == {1, 2}


def test_load_shards_survives_a_corrupt_shard(tmp_path):
    """A shard truncated by a kill must not take down the whole resume."""
    flush([{"label": "transit", "seed": 1}], str(tmp_path), 0)
    (tmp_path / "part_0001.parquet").write_bytes(b"not a parquet file")
    df = load_shards(str(tmp_path))
    assert len(df) == 1, "should skip the corrupt shard and keep the good one"


def test_load_shards_on_empty_dir(tmp_path):
    assert load_shards(str(tmp_path)).empty


def test_finish_deduplicates_overlapping_shards(tmp_path, capsys):
    """Re-running can recompute a seed; the merge must not double-count it."""
    flush([{"label": "transit", "seed": 1, "detected": True}], str(tmp_path), 0)
    flush([{"label": "transit", "seed": 1, "detected": True},
           {"label": "noise", "seed": 2, "detected": False}], str(tmp_path), 1)
    out = tmp_path / "train.parquet"
    finish(load_shards(str(tmp_path)), str(out))
    df = pd.read_parquet(out)
    assert len(df) == 2, f"expected 2 unique rows, got {len(df)}"


def test_resume_skips_completed_work(tmp_path):
    """End-to-end: a second run must compute nothing and still emit the file."""
    out = tmp_path / "train.parquet"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "make_dataset.py")
    # --source synthetic: this test exercises the shard/resume machinery, not
    # real-background injection, and must not depend on a downloaded baseline bank
    cmd = [sys.executable, script, "-n", "1", "-o", str(out), "-j", "2",
           "--source", "synthetic"]

    first = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert first.returncode == 0, first.stderr
    n_first = len(pd.read_parquet(out))
    assert n_first > 0

    second = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert second.returncode == 0, second.stderr
    assert "resuming" in second.stdout
    assert len(pd.read_parquet(out)) == n_first, "resume changed the row count"
