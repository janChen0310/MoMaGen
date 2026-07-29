# tests/test_mjpeg_stream.py
"""Tests for the extracted MJPEG/keyboard HTTP layer used by the web-teleop scripts.

This module exists because `momagen/scripts/collect_source_web_teleop.py` imports
`omnigibson` at module scope and runs its server + sim loop at import time, so it
cannot be imported or exercised in a plain test venv (no simulator, no GPU). The
HTTP/streaming layer was pulled out into `momagen/utils/mjpeg_stream.py` -- pure
stdlib plus a caller-supplied encode callable -- specifically so it can be driven
over real TCP sockets here.

Three defects are pinned:
  1. HTTP/1.0-with-no-keep-alive -- every one of the 20/sec `/key` polls paid a
     fresh TCP handshake. Fixed by `protocol_version = "HTTP/1.1"` plus correct
     framing (Content-Length, or Connection: close for /stream). Pinned by
     `test_key_endpoint_serves_two_requests_over_one_keepalive_connection`.
  2. Blocking MJPEG writes into the OS's ~4MB default send buffer let stale frames
     queue up for a slow client. Pinned by `test_publish_keeps_only_the_newest_frame`
     (no backlog in FrameBuffer itself) and
     `test_stream_survives_a_client_that_stops_reading` (the handler must not wedge
     the server -- a fresh, unrelated request must still succeed while a stalled
     stream sits unread).
  3. Render + JPEG encode ran inline in the sim's `env.step()` loop. Pinned by
     `test_encode_worker_runs_encode_off_the_publishing_thread` -- publish_raw()
     must return immediately even though the (deliberately slow) encode function
     it hands off to has not finished.

All tests use real ephemeral-port TCP sockets (`http.client`/`socket`), not mocks,
per the lesson from the WebXR task: a `stop()`/`shutdown()` that silently no-ops
on the installed server was only ever caught by a real start/stop/start cycle.
"""
import http.client
import os
import socket
import threading
import time

import pytest

from momagen.utils.mjpeg_stream import FrameBuffer, make_handler, serve


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _start(frame_buffer=None, key_sink=None, html="<html><body>hi</body></html>"):
    """Build a handler + start a server on an ephemeral loopback port."""
    fb = frame_buffer if frame_buffer is not None else FrameBuffer()
    sink = key_sink if key_sink is not None else (lambda k: None)
    handler_cls = make_handler(fb, sink, html)
    server = serve(handler_cls, host="127.0.0.1", port=0)
    port = server.server_address[1]
    assert port != 0, "serve() must record the actually-bound ephemeral port"
    return fb, server, port


# ---------------------------------------------------------------------------
# Defect 1: HTTP/1.0 + no keep-alive -> a fresh TCP handshake per /key poll
# ---------------------------------------------------------------------------

def test_key_endpoint_serves_two_requests_over_one_keepalive_connection():
    received = []
    fb, server, port = _start(key_sink=received.append)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)

        conn.request("GET", "/key?k=w")
        r1 = conn.getresponse()
        assert r1.status == 204
        assert r1.version == 11, "must advertise HTTP/1.1"
        r1.read()

        # Second request on the SAME connection, no reconnect: if keep-alive were
        # broken (HTTP/1.0 default, or a framing bug that confuses the client
        # about where the first response ends) this raises instead of cleanly
        # returning 204.
        conn.request("GET", "/key?k=s")
        r2 = conn.getresponse()
        assert r2.status == 204
        assert r2.version == 11
        r2.read()

        conn.close()
        assert received == ["w", "s"]
    finally:
        server.shutdown()


def test_key_endpoint_ignores_missing_k_but_still_returns_204():
    received = []
    fb, server, port = _start(key_sink=received.append)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/key")
        resp = conn.getresponse()
        assert resp.status == 204
        resp.read()
        conn.close()
        assert received == []
    finally:
        server.shutdown()


def test_index_page_has_correct_content_length_over_http11():
    html = "<html><body>hello teleop</body></html>"
    fb, server, port = _start(html=html)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.version == 11
        body = resp.read()
        assert body.decode() == html
        # HTTP/1.1 framing is unforgiving: a missing/incorrect Content-Length
        # hangs the client waiting for a body that never ends.
        assert resp.getheader("Content-Length") == str(len(html.encode()))
        conn.close()

        # Any other path (not /key, not /stream) also serves the HTML, same as
        # the original inline handler's fallthrough branch.
        conn2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn2.request("GET", "/whatever")
        resp2 = conn2.getresponse()
        assert resp2.status == 200
        assert resp2.read().decode() == html
        conn2.close()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Defect 2: unbounded backlog behind a slow client
# ---------------------------------------------------------------------------

def test_publish_keeps_only_the_newest_frame():
    fb = FrameBuffer()
    for i in range(500):
        fb.publish(("frame-%d" % i).encode())
    frame_id, data = fb.latest()
    assert data == b"frame-499", "only the newest published frame may survive"
    assert frame_id == 500, "frame_id must advance once per publish, not batch/drop silently"


def test_stream_survives_a_client_that_stops_reading():
    # NOTE: because ThreadingHTTPServer already isolates every connection on its
    # own thread, "an unrelated request still succeeds" is true even WITHOUT the
    # fix (the stalled connection's thread just blocks forever on its own,
    # unnoticed). That alone would not be a discriminating regression test, so
    # the actual pin here is that the stalled connection's OWN handler thread
    # must terminate within a bounded window instead of blocking in write()
    # forever -- proven by tracking the specific thread object that appears
    # when the stalled client connects.
    fb = FrameBuffer()
    fb.publish(os.urandom(200_000))
    fb, server, port = _start(frame_buffer=fb)
    stop_feeding = threading.Event()

    def feeder():
        while not stop_feeding.is_set():
            fb.publish(os.urandom(200_000))
            time.sleep(0.005)

    feeder_thread = threading.Thread(target=feeder, daemon=True, name="test-feeder")
    feeder_thread.start()
    stalled = None
    try:
        before = set(threading.enumerate())

        # Connect to /stream and never read a single byte back -- this is the
        # slow/stalled client. With the OS's default ~4MB send buffer and no
        # write timeout, the per-connection handler thread would block in
        # write() indefinitely (queued_bytes / bandwidth lag, plus a leaked
        # thread). Bounding SO_SNDBUF + a socket timeout must make it give up.
        stalled = socket.create_connection(("127.0.0.1", port), timeout=5)
        stalled.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n")

        new_threads = set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            new_threads = set(threading.enumerate()) - before
            if new_threads:
                break
            time.sleep(0.01)
        assert new_threads, "stream handler thread for the stalled client never started"

        # This is the actual regression pin: that thread must die on its own
        # within a bounded window rather than blocking in write() forever. This
        # loop is itself bounded, so a regression fails fast instead of hanging
        # the suite.
        deadline = time.monotonic() + 3
        still_alive = new_threads
        while time.monotonic() < deadline:
            still_alive = [t for t in new_threads if t.is_alive()]
            if not still_alive:
                break
            time.sleep(0.05)
        assert not still_alive, (
            "the stalled client's stream handler thread never terminated -- "
            "it is blocking in write() instead of giving up on a slow client"
        )

        # Concretely, the server as a whole is still usable: an unrelated
        # request on a fresh connection completes normally.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/key?k=w")
        resp = conn.getresponse()
        assert resp.status == 204
        resp.read()
        conn.close()
    finally:
        stop_feeding.set()
        feeder_thread.join(timeout=2)
        if stalled is not None:
            stalled.close()
        server.shutdown()


def test_stream_delivers_multipart_jpeg_frames_to_a_reading_client():
    fb = FrameBuffer()
    fb.publish(b"\xff\xd8\xff\xd9fake-jpeg-bytes")
    fb, server, port = _start(frame_buffer=fb)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "multipart/x-mixed-replace" in resp.getheader("Content-Type", "")
        chunk = resp.read(64)
        assert chunk, "expected the first multipart frame to arrive promptly"
        assert b"--frame" in chunk
        conn.close()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Defect 3: render+encode ran inline on the publishing (sim) thread
# ---------------------------------------------------------------------------

def test_encode_worker_runs_encode_off_the_publishing_thread():
    fb = FrameBuffer()
    encode_started = threading.Event()
    release_encode = threading.Event()
    encoded_marker = b"THE-ENCODED-FRAME"

    def slow_encode(raw):
        encode_started.set()
        assert release_encode.wait(timeout=5), "test setup deadlocked"
        return encoded_marker

    worker = threading.Thread(target=fb.encode_worker, args=(slow_encode,), daemon=True)
    worker.start()
    try:
        t0 = time.monotonic()
        fb.publish_raw(object())  # stand-in for a raw rendered frame
        publish_duration = time.monotonic() - t0
        assert publish_duration < 0.2, (
            "publish_raw() must return immediately -- it must not run the encode "
            "itself on the publishing (sim) thread"
        )

        assert encode_started.wait(timeout=2), "encode_worker never picked up the raw frame"

        # The slow encode is still blocked inside slow_encode() at this point, so
        # the encoded frame must not have appeared yet.
        _, jpeg_before = fb.latest()
        assert jpeg_before != encoded_marker

        release_encode.set()

        deadline = time.monotonic() + 2
        jpeg_after = None
        while time.monotonic() < deadline:
            _, jpeg_after = fb.latest()
            if jpeg_after == encoded_marker:
                break
            time.sleep(0.005)
        assert jpeg_after == encoded_marker, "encoded frame never appeared after release"
    finally:
        fb.stop()
        worker.join(timeout=2)


def test_encode_worker_only_encodes_the_newest_raw_frame():
    # Mirrors defect 2 but for the raw (pre-encode) side: many rapid publish_raw()
    # calls made *before* any worker is draining must not force every one of them
    # through the encoder -- only the latest survives to be encoded.
    fb = FrameBuffer()
    calls = []
    done = threading.Event()

    def fast_encode(raw):
        calls.append(raw)
        done.set()
        return b"ok"

    for i in range(1000):
        fb.publish_raw(i)

    worker = threading.Thread(target=fb.encode_worker, args=(fast_encode,), daemon=True)
    worker.start()
    try:
        assert done.wait(timeout=2), "encode_worker never encoded anything"
        # Give a moment to make sure it doesn't also chew through backlog.
        time.sleep(0.1)
        assert calls == [999], "expected exactly one encode call, for the newest raw frame"
    finally:
        fb.stop()
        worker.join(timeout=2)


class _PublishDuringSwapLock:
    """A `threading.Lock` stand-in that publishes ONE extra raw frame the instant
    `encode_worker` releases the lock it took to swap `_raw` out.

    This is deliberately white-box (it replaces `FrameBuffer._lock`) because the
    lost-wakeup window it targets -- between the swap and `_raw_ready.clear()` --
    has no public seam, and a `sleep`-and-hope test for a race is worthless in CI.
    Injecting at the lock's release point makes the interleaving EXACT: the extra
    `publish_raw()` lands after the swap and, in the buggy ordering, immediately
    before the `clear()` that erases its wakeup.

    It fires only on a post-swap release (`_raw is None` on exit); `publish_raw()`
    and `publish()` leave other state, so they cannot trigger it.
    """

    def __init__(self, frame_buffer, frame):
        self._inner = threading.Lock()
        self._fb = frame_buffer
        self._frame = frame
        self.fired = threading.Event()

    def __enter__(self):
        return self._inner.__enter__()

    def __exit__(self, *exc):
        result = self._inner.__exit__(*exc)
        if not self.fired.is_set() and self._fb._raw is None:
            self.fired.set()
            # Released above, so this re-acquire cannot deadlock.
            self._fb.publish_raw(self._frame)
        return result


def test_encode_worker_does_not_lose_a_frame_published_during_the_swap():
    # Regression pin for a lost-wakeup freeze: `_raw_ready.clear()` used to run
    # AFTER the swap, so a frame published in between erased its own wakeup. The
    # worker then only timed out (it never re-reads `_raw` on the timeout path)
    # and the MJPEG stream stayed stuck on a stale frame for as long as nothing
    # else was published -- i.e. forever, once the sim loop stopped or paused.
    fb = FrameBuffer()
    encoded = []
    second_encoded = threading.Event()

    def encode(raw):
        encoded.append(raw)
        if raw == "second":
            second_encoded.set()
        return b"jpeg"

    fb._lock = _PublishDuringSwapLock(fb, "second")
    fb.publish_raw("first")

    worker = threading.Thread(target=fb.encode_worker, args=(encode,), daemon=True)
    worker.start()
    try:
        assert fb._lock.fired.wait(timeout=5), "the injected publish never ran"
        # Bounded wait, so a regression fails fast instead of hanging the suite.
        assert second_encoded.wait(timeout=5), (
            "a frame published between the swap and _raw_ready.clear() was never "
            "encoded -- clear() must happen BEFORE the swap (or the timeout path "
            "must re-check _raw), otherwise the stream freezes on a stale frame"
        )
        assert encoded[:2] == ["first", "second"]
    finally:
        fb.stop()
        worker.join(timeout=2)


# ---------------------------------------------------------------------------
# serve()/shutdown() lifecycle: port must be genuinely released, not just
# "the thread is a daemon" (the Werkzeug stop()-that-silently-no-ops lesson).
# ---------------------------------------------------------------------------

def test_serve_shutdown_start_cycle_releases_the_port():
    handler_cls = make_handler(FrameBuffer(), lambda k: None, "<html></html>")
    first = serve(handler_cls, host="127.0.0.1", port=0)
    port = first.server_address[1]
    assert port != 0

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/")
    resp = conn.getresponse()
    assert resp.status == 200
    resp.read()
    conn.close()

    first.shutdown()

    # The port must be genuinely released, not just "the thread is a daemon".
    with pytest.raises(OSError):
        c = socket.create_connection(("127.0.0.1", port), timeout=2)
        c.close()

    second = serve(handler_cls, host="127.0.0.1", port=port)
    try:
        conn2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn2.request("GET", "/")
        resp2 = conn2.getresponse()
        assert resp2.status == 200
        resp2.read()
        conn2.close()
    finally:
        second.shutdown()


def test_shutdown_is_safe_to_call_twice():
    handler_cls = make_handler(FrameBuffer(), lambda k: None, "<html></html>")
    server = serve(handler_cls, host="127.0.0.1", port=0)
    server.shutdown()
    server.shutdown()  # must not raise
