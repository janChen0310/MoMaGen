# momagen/utils/webxr_server.py
"""WebXR phone-teleop Socket.IO/Flask server.

Ported from tidybot_ros src/tidybot_policy/tidybot_policy/phone_teleop_server.py.
Only the `WebServer` class survives the port: the `WebServerPublisher` rclpy node
and every rclpy / tidybot_utils / ament_index_python import are deleted. This module
targets OmniGibson, which has no ROS integration at all, not the tidybot_ros ROS
graph, so it must import cleanly with no ROS installed.

Messages received from the phone are pushed onto a `collections.deque` for a
simulator loop to drain, exactly like the `key_queue` the keyboard-teleop path
already drains today.

Lifecycle is explicit: `WebServer(...)` builds the app, the handlers and the queue
without touching the network, and `start()` / `stop()` bind and release the port.
Constructing must stay side-effect free — it is the only reason the socket handlers
can be exercised by tests (and by a caller doing a dry run) with nothing listening.
"""
import logging
import os
import socket
import threading
from collections import deque

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from werkzeug.serving import make_server

GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"

# momagen/assets/webxr holds the vendored phone client (index.html, webxr-button.js,
# socket.io.min.js). template_folder and static_folder both point here so that
# render_template("index.html", ...) and the /static/* asset requests it makes
# (webxr-button.js, socket.io.min.js) resolve to the same vendored directory.
_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "webxr"
)

# Default bound for the received-message deque. For teleop only the newest pose
# matters, so a stalled consumer (e.g. sim loop paused/crashed) should drop the
# oldest stale message rather than grow memory without bound.
_DEFAULT_QUEUE_MAXLEN = 100

_DEFAULT_PORT = 5000


class WebServer:
    """Flask + Socket.IO server that serves the phone webapp and queues its messages.

    `queue` is a `collections.deque`; every WebXR message received over the socket
    is appended to it for a simulator loop to `popleft()`, mirroring how `key_queue`
    is drained today for the keyboard-teleop fallback. If no `queue` is supplied, a
    bounded one (`maxlen=_DEFAULT_QUEUE_MAXLEN`) is created automatically; pass one
    explicitly to choose a different bound (or, deliberately, an unbounded deque).

    Usage::

        server = WebServer(queue)
        server.start()          # binds `port`, serves in a daemon thread
        ...                     # sim loop drains `server.queue`
        server.stop()           # releases the port

    `port=0` asks the OS for an ephemeral port; `start()` writes the real one back
    to `self.port`.
    """

    def __init__(self, queue: deque = None, record_enabled: bool = False,
                 assets_dir: str = _ASSETS_DIR, host: str = "0.0.0.0",
                 port: int = _DEFAULT_PORT):
        self.app = Flask(
            __name__,
            template_folder=assets_dir,
            static_folder=assets_dir,
            static_url_path="/static",
        )
        # Use threading async mode with the Werkzeug server; requires simple-websocket
        # installed for WS. init_app wraps app.wsgi_app in the Socket.IO middleware, so
        # serving `self.app` through werkzeug serves the socket transport too.
        self.socketio = SocketIO(self.app, async_mode="threading", cors_allowed_origins="*")
        self.queue = queue if queue is not None else deque(maxlen=_DEFAULT_QUEUE_MAXLEN)
        self.address = None
        self.host = host
        self.port = port
        self.server_thread = None
        self.record_enabled = record_enabled
        self._server = None

        @self.app.route("/")
        def index():
            return render_template("index.html", record_enabled=self.record_enabled)

        @self.socketio.on("message")
        def handle_message(data):
            # Send the timestamp back for RTT calculation (expected RTT on 5 GHz Wi-Fi
            # is 7 ms) -- this is how the operator confirms the link is healthy.
            # `.get` rather than `data["timestamp"]`: a message without one is not a
            # reason to raise out of the handler and drop the pose it carries.
            timestamp = data.get("timestamp") if isinstance(data, dict) else None
            if timestamp is not None:
                emit("echo", timestamp)

            # Push data onto the deque for the sim loop to drain.
            self.queue.append(data)

        # Explicit handlers for save/discard events
        @self.socketio.on("save_episode")
        def handle_save():
            if self.record_enabled:
                self.queue.append({"state_update": "save_episode", "timestamp": 0})

        @self.socketio.on("discard_episode")
        def handle_discard():
            if self.record_enabled:
                self.queue.append({"state_update": "discard_episode", "timestamp": 0})

        # Reduce verbose Flask log output
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def _discover_address(self):
        """Best-effort LAN IP, so the printed URL is one the phone can actually reach."""
        # Only guess when bound to a wildcard; an explicit host IS the answer, and
        # printing the LAN IP for a 127.0.0.1-bound server would be a lie.
        if self.host not in ("0.0.0.0", "::", ""):
            self.address = self.host
            return self.address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("8.8.8.8", 1))
            self.address = s.getsockname()[0]
        except Exception:
            self.address = "127.0.0.1"
        finally:
            s.close()
        return self.address

    def start(self):
        """Bind the port and serve in a daemon thread. Returns self.

        `make_server` is used instead of `socketio.run()` specifically so that the
        underlying server OBJECT is retained: it is the only handle on Werkzeug 3.x
        that can actually stop the server. The `request.environ["werkzeug.server.
        shutdown"]` hook the previous `/__shutdown__` route relied on was REMOVED in
        Werkzeug 2.1, so that route returned 500 forever and `stop()` was a no-op.
        """
        if self._server is not None:
            raise RuntimeError("WebServer.start() called on an already-started server")
        # threaded=True mirrors what socketio.run()/run_simple do in threading async
        # mode; each websocket occupies a connection for its lifetime, so a
        # single-threaded server would serve exactly one phone and then wedge.
        self._server = make_server(self.host, self.port, self.app, threaded=True)
        # Records the real port when self.port was 0 (ephemeral).
        self.port = self._server.port
        self.server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="webxr-server")
        self.server_thread.start()
        self._discover_address()
        print(f"Serving WebXR teleop client at "
              f"{BOLD}{GREEN}http://{self.address}:{self.port}{RESET}")
        return self

    def stop(self):
        """Stop serving and release the port. Safe before start() and safe twice."""
        server, self._server = self._server, None
        if server is not None:
            # shutdown() ends serve_forever; server_close() closes the LISTENING
            # socket. Without the second call the port stays claimed and the next
            # WebServer on it dies with "Address already in use".
            server.shutdown()
            server.server_close()
        thread, self.server_thread = self.server_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
