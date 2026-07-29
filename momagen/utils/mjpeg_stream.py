# momagen/utils/mjpeg_stream.py
"""MJPEG-over-HTTP + 20Hz keyboard-poll HTTP layer for the web-teleop scripts.

Extracted out of `momagen/scripts/collect_source_web_teleop.py`, which imports
`omnigibson` at module scope and runs its server + sim loop at import time, so it
cannot be imported or tested without a simulator. This module is pure stdlib plus
a caller-supplied encode callable, so it imports cleanly (and is testable) with no
omnigibson/torch/PIL/cv2 anywhere in the process.

It fixes three latency defects present in the inline version:

  1. HTTP/1.0 with no keep-alive: `BaseHTTPRequestHandler` defaults to HTTP/1.0,
     so every one of the 20/sec `/key` polls (one per held key, from the browser's
     `setInterval`) paid a fresh TCP handshake (~40 ms) and competed with the MJPEG
     stream for the browser's ~6-connections-per-host budget. Fixed by setting
     `protocol_version = "HTTP/1.1"` on the handler produced by `make_handler`, with
     correct framing (`Content-Length` on `/` and `/key`; `Connection: close` on the
     open-ended `/stream` body, since HTTP/1.1 requires unambiguous framing --  a
     missing/incorrect length hangs the client).

  2. Blocking MJPEG writes into the OS's ~4 MB default socket send buffer: at a
     5-11 Mbps encode rate, a client that falls behind lets `queued_bytes /
     bandwidth` (potentially seconds) of STALE frames sit in the kernel buffer --
     the app-level "send only the newest frame" check can't help, because the
     stall is *inside* `write()`, not in what gets selected to send. Fixed by
     bounding `SO_SNDBUF` to roughly a couple of frames, giving the stream socket a
     short write timeout, and treating a write timeout as "drop this client's
     stream" rather than blocking (see `_serve_stream` below). `FrameBuffer` itself
     is newest-frame-only on both its raw and encoded slots, so there is never an
     internal backlog to begin with.

  3. Render + JPEG encode inline on the sim's critical path: `ann.get_data()` (a
     GPU->CPU readback) plus the PIL/cv2 encode used to run inside the `env.step()`
     loop, so encode time was added to every simulation step. Fixed by
     `FrameBuffer.encode_worker`, which runs the caller-supplied encode function on
     its own thread, off whatever thread calls `publish_raw()`.
"""
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Bound on the stream socket's kernel send buffer: roughly 1-2 JPEG frames at the
# resolutions/quality this teleop uses, so a slow client can only have that much
# STALE data queued up in the kernel before write() either succeeds or times out.
_STREAM_SNDBUF_BYTES = 65536

# Per-write timeout on the stream socket. On timeout we give up on that specific
# client's stream rather than block -- "drop frames on a slow client instead of
# blocking". Short enough that a stalled client is detected quickly, long enough
# not to false-positive under normal jitter.
_STREAM_WRITE_TIMEOUT_S = 0.3

# Poll interval while waiting for a new frame_id (mirrors the original 4ms poll).
_STREAM_POLL_INTERVAL_S = 0.004

# Minimum spacing between frames actually written to the client, so a fast
# publisher cannot exceed a sane send rate even for a client that reads fine.
_STREAM_MIN_FRAME_INTERVAL_S = 1.0 / 30.0

# Default poll interval for encode_worker's wait on a new raw frame.
_ENCODE_POLL_INTERVAL_S = 0.01


class FrameBuffer:
    """Thread-safe, newest-frame-only holder, split into a raw slot and a
    servable (encoded) slot.

    - `publish_raw(frame)` is meant to be called every simulation step, from the
      sim thread. It is O(1) and never blocks: it just overwrites whatever raw
      frame was previously pending, so a publisher that outruns the encoder never
      builds an unbounded backlog (defect 2) and never runs the encoder itself
      (defect 3).
    - `encode_worker(encode_fn)` is meant to run on its own dedicated thread. It
      drains the newest pending raw frame and calls `encode_fn(frame)` -- which
      may be arbitrarily slow (a real JPEG encode is a GPU->CPU readback plus a
      PIL/cv2 encode) -- entirely off the thread that calls `publish_raw()`. The
      result is handed to `publish()`.
    - `publish(jpeg_bytes)` stores an already-encoded frame as the newest servable
      frame (bumping `frame_id`); a caller with pre-encoded bytes and no need for
      the async path may call this directly.
    - `latest()` returns `(frame_id, jpeg_bytes)`, read by the HTTP stream
      handler; `jpeg_bytes` is `None` until the first frame is published.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._raw = None
        self._raw_ready = threading.Event()
        self._frame_id = 0
        self._jpeg = None
        self._stop = threading.Event()
        self._encode_failed = False

    def publish_raw(self, frame):
        """Hand off a raw (unencoded) frame. Never blocks, never encodes."""
        with self._lock:
            self._raw = frame
        self._raw_ready.set()

    def publish(self, jpeg_bytes):
        """Store an already-encoded frame as the newest servable frame."""
        with self._lock:
            self._jpeg = jpeg_bytes
            self._frame_id += 1

    def latest(self):
        with self._lock:
            return self._frame_id, self._jpeg

    def encode_worker(self, encode_fn, poll_interval=_ENCODE_POLL_INTERVAL_S):
        """Run loop: drains the newest raw frame and encodes it off the caller's
        (i.e. the publishing/sim) thread. Intended to be run as a thread target:

            threading.Thread(target=frame_buffer.encode_worker, args=(enc,),
                              daemon=True).start()

        Exits once `stop()` is called; safe to run in a daemon thread regardless.
        """
        while not self._stop.is_set():
            if not self._raw_ready.wait(timeout=poll_interval):
                continue
            # clear() BEFORE the swap, never after. `publish_raw()` writes `_raw`
            # and only THEN sets the event, so a clear() placed after the swap can
            # erase the wakeup belonging to a frame that is already sitting in
            # `_raw` (published in the window between the swap and the clear).
            # Nothing re-reads `_raw` on the timeout path, so that frame -- and
            # every frame after it, once publishing stops -- is never encoded and
            # the stream freezes on a stale frame. Clearing first cannot lose a
            # frame: the worst case is a spurious wakeup whose swap yields None,
            # handled by the `raw is None` guard below.
            self._raw_ready.clear()
            with self._lock:
                raw, self._raw = self._raw, None
            if raw is None:
                continue
            try:
                self.publish(encode_fn(raw))
            except Exception as exc:
                # A dead encode worker means the stream silently freezes forever with
                # only a buried traceback -- far worse than dropping one frame. Log the
                # first failure, then keep serving; a transient or per-frame encoder
                # problem must not take the video down permanently.
                if not self._encode_failed:
                    self._encode_failed = True
                    print("mjpeg_stream: encode failed (%s: %s); dropping frames, stream stays up"
                          % (type(exc).__name__, exc), flush=True)

    def stop(self):
        """Ask a running `encode_worker` loop to exit. Safe to call any time."""
        self._stop.set()


def make_handler(frame_buffer, key_sink, html):
    """Build a `BaseHTTPRequestHandler` subclass serving:

      - `/`       -- the given HTML page (any path other than /key* or /stream
                     also falls through to this, matching the original handler).
      - `/key`    -- reads `k` from the query string and, if present, calls
                     `key_sink(k)`; always responds 204. `key_sink` is any
                     callable taking a single string (e.g. `deque.append`).
      - `/stream` -- an MJPEG (`multipart/x-mixed-replace`) push of
                     `frame_buffer`'s newest encoded frame.

    `protocol_version = "HTTP/1.1"` (defect 1): every non-streaming response below
    sends an accurate `Content-Length`; `/stream`, whose length is unknowable in
    advance, sends `Connection: close` instead so the framing stays unambiguous.
    """
    html_bytes = html.encode("utf-8") if isinstance(html, str) else html

    class MjpegTeleopHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass  # keep collection sessions quiet; matches the original handler

        def _send_complete(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/key"):
                self._handle_key()
            elif self.path == "/stream":
                self._serve_stream()
            else:
                self._send_complete(200, html_bytes, "text/html; charset=utf-8")

        def _handle_key(self):
            k = parse_qs(urlparse(self.path).query).get("k", [""])[0]
            if k:
                key_sink(k)
            # 204 No Content: RFC 7231 forbids a body, so (matching
            # BaseHTTPRequestHandler's own send_error convention) no
            # Content-Length is sent either -- http.client and every real HTTP
            # client know a 204 has no body regardless.
            self.send_response(204)
            self.end_headers()

        def _serve_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            # The body length is unknowable up front (it's a push that runs
            # until the client goes away), so HTTP/1.1 framing requires this be
            # explicit: Connection: close tells both the client and
            # BaseHTTPRequestHandler's own keep-alive bookkeeping that this
            # connection is a one-shot, ending only when the body ends.
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _STREAM_SNDBUF_BYTES)
            except OSError:
                pass  # best-effort; some platforms/sockets don't allow shrinking it
            self.connection.settimeout(_STREAM_WRITE_TIMEOUT_S)

            last_id = -1
            try:
                while True:
                    frame_id, jpeg = frame_buffer.latest()
                    if jpeg is not None and frame_id != last_id:
                        last_id = frame_id
                        header = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)
                        # A write that would otherwise block indefinitely on a
                        # slow/stalled client instead raises after
                        # _STREAM_WRITE_TIMEOUT_S; we give up on this one
                        # connection rather than let it (or the queued-up stale
                        # bytes behind it) accumulate lag.
                        self.wfile.write(header)
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(_STREAM_MIN_FRAME_INTERVAL_S)  # cap the send rate
                    else:
                        time.sleep(_STREAM_POLL_INTERVAL_S)
            except (TimeoutError, OSError):
                # Client stalled or disconnected: drop this stream. Connection:
                # close above already told the framework not to try to read a
                # further request off this same socket.
                return

    return MjpegTeleopHandler


def serve(handler_cls, host="0.0.0.0", port=0):
    """Start `handler_cls` on a `ThreadingHTTPServer` bound to (host, port) in a
    daemon thread; returns the started server. `port=0` binds an ephemeral port,
    discoverable afterward via `server.server_address[1]`.

    The returned server's `shutdown()` is patched to genuinely release the port
    (`shutdown()` to stop `serve_forever`, then `server_close()` to release the
    listening socket, then join the serving thread) -- see the Werkzeug lesson
    from the WebXR task: a `stop()`/`shutdown()` that only stops the loop but
    never closes the socket leaves the port claimed, so the NEXT server on it
    dies with "Address already in use". Safe to call twice.
    """
    server = ThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="mjpeg-http")
    thread.start()

    base_shutdown = server.shutdown
    _shutdown_called = threading.Event()

    def _shutdown():
        if _shutdown_called.is_set():
            return
        _shutdown_called.set()
        base_shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    server.shutdown = _shutdown
    return server
