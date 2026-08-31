"""The download harness must survive a hang, not just an exception.

`with_backoff` only ever saw *raised* failures.  A stalled socket raises
nothing, so a hang was invisible to it -- that is what wedged a full run at
95% completion.  These tests reproduce a hang deliberately rather than
asserting the fix from its shape, because the failure mode is precisely the one
that looks fine in code review.
"""
from __future__ import annotations

import os
import socket
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exodet.fetch import (DownloadTimeout, call_with_deadline,   # noqa: E402
                          install_socket_timeout, with_backoff)


def test_deadline_fires_on_a_sleeping_call():
    def hangs():
        time.sleep(30)
        return "never"

    t0 = time.monotonic()
    with pytest.raises(DownloadTimeout):
        call_with_deadline(hangs, timeout=0.5)
    assert time.monotonic() - t0 < 5.0, "deadline did not actually abandon the call"


def test_deadline_is_transparent_when_the_call_returns():
    assert call_with_deadline(lambda x: x * 2, 21, timeout=5.0) == 42


def test_deadline_propagates_the_original_exception():
    def boom():
        raise ValueError("upstream failure")

    with pytest.raises(ValueError, match="upstream failure"):
        call_with_deadline(boom, timeout=5.0)


def test_backoff_retries_a_hang_through_the_same_path_as_an_exception():
    """The whole point of subclassing TimeoutError: one retry path, not two."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            return call_with_deadline(time.sleep, 30, timeout=0.3)
        return "recovered"

    assert with_backoff(flaky, attempts=4, base=1.0, cap=0.05) == "recovered"
    assert calls["n"] == 3, "a hang was not retried like an ordinary failure"


def test_unroutable_address_is_bounded_by_the_socket_layer():
    """TEST-NET-1 (RFC 5737) is guaranteed unroutable, so a connect to it
    stalls exactly the way a wedged MAST socket does: no RST, no SYN-ACK, just
    silence.  Here the socket timeout is the *tighter* of the two layers, so it
    is the one that fires -- and it raises an ordinary TimeoutError, which
    `with_backoff` already retries.  That is the layer covering library code
    (astroquery, lightkurve) that exposes no timeout parameter to pass down.
    """
    install_socket_timeout()
    assert socket.getdefaulttimeout() is not None, "no process-wide socket timeout"

    def connect():
        s = socket.socket()
        try:
            s.settimeout(1.0)
            s.connect(("192.0.2.1", 80))       # blackholed
        finally:
            s.close()

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        call_with_deadline(connect, timeout=20.0)   # deliberately the looser layer
    assert time.monotonic() - t0 < 10.0, "connect was not bounded at all"


def test_unroutable_address_is_bounded_by_the_deadline_when_the_socket_is_not():
    """The case the socket timeout alone cannot cover.

    A socket timeout bounds each individual read, never the total.  With no
    socket timeout set, a blackholed connect blocks for the OS-level TCP retry
    budget -- minutes on Linux/macOS -- which is exactly the stall that wedged
    the last run.  The wall-clock deadline is the only layer that bounds it.
    """
    def connect_untimed():
        s = socket.socket()
        try:
            s.settimeout(None)                  # no per-read bound at all
            s.connect(("192.0.2.1", 80))
        finally:
            s.close()

    t0 = time.monotonic()
    with pytest.raises(DownloadTimeout):
        call_with_deadline(connect_untimed, timeout=2.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 6.0, f"deadline did not bound an untimed connect ({elapsed:.1f}s)"
