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


def _blackhole_server():
    """A local listener that accepts a connection and then never sends a byte.

    This replaces the earlier TEST-NET-1 (192.0.2.1) tests. Reserved
    documentation space is *supposed* to be unroutable, but it is not reliably
    so: on the network these tests were re-run against, 192.0.2.1:80 accepted a
    connection in 0.25 s -- something upstream (captive portal, ISP interceptor)
    answers for it. The tests then failed with DID NOT RAISE, reporting a broken
    timeout when the timeout was fine and the *premise* was wrong.

    A loopback socket that accepts and stalls reproduces the real failure mode --
    connection established, no data ever arrives -- and does so identically on
    every network.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def test_a_stalled_read_is_bounded_by_the_socket_layer():
    """Connection accepted, no data ever sent: the process-wide timeout fires."""
    srv, port = _blackhole_server()
    try:
        install_socket_timeout(1.0)
        assert socket.getdefaulttimeout() is not None, "no process-wide socket timeout"

        t0 = time.monotonic()
        with pytest.raises((TimeoutError, socket.timeout, OSError)):
            with socket.create_connection(("127.0.0.1", port)) as s:
                s.recv(1)                      # server never writes
        assert time.monotonic() - t0 < 10.0, "the stalled read was not bounded"
    finally:
        srv.close()
        socket.setdefaulttimeout(None)


def test_a_stalled_read_is_bounded_by_the_deadline_when_the_socket_is_not():
    """With no socket timeout at all, the wall-clock deadline must still fire.

    This is the case a per-read timeout cannot cover on its own, and the reason
    the harness carries two layers rather than one.
    """
    srv, port = _blackhole_server()
    socket.setdefaulttimeout(None)             # deliberately remove layer one
    try:
        def stalls():
            with socket.create_connection(("127.0.0.1", port)) as s:
                return s.recv(1)

        t0 = time.monotonic()
        with pytest.raises(DownloadTimeout):
            call_with_deadline(stalls, timeout=1.0)
        assert time.monotonic() - t0 < 10.0, "the deadline did not abandon the call"
    finally:
        srv.close()
        socket.setdefaulttimeout(None)
