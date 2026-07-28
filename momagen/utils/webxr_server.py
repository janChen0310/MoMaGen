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
"""
import logging
import os
import socket
import threading
import time
from collections import deque
from urllib.request import Request, urlopen

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

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


class WebServer:
    """Flask + Socket.IO server that serves the phone webapp and queues its messages.

    `queue` is a `collections.deque`; every WebXR message received over the socket
    is appended to it for a simulator loop to `popleft()`, mirroring how `key_queue`
    is drained today for the keyboard-teleop fallback. If no `queue` is supplied, a
    bounded one (`maxlen=_DEFAULT_QUEUE_MAXLEN`) is created automatically; pass one
    explicitly to choose a different bound (or, deliberately, an unbounded deque).
    """

    def __init__(self, queue: deque = None, record_enabled: bool = False,
                 assets_dir: str = _ASSETS_DIR):
        self.app = Flask(
            __name__,
            template_folder=assets_dir,
            static_folder=assets_dir,
            static_url_path="/static",
        )
        # Use threading async mode with Werkzeug dev server; requires simple-websocket
        # installed for WS.
        self.socketio = SocketIO(self.app, async_mode="threading", cors_allowed_origins="*")
        self.queue = queue if queue is not None else deque(maxlen=_DEFAULT_QUEUE_MAXLEN)
        self.address = None
        self.port = 5000
        self.server_thread = None
        self.record_enabled = record_enabled

        @self.app.route("/")
        def index():
            return render_template("index.html", record_enabled=self.record_enabled)

        @self.app.route("/__shutdown__")
        def shutdown():
            func = request.environ.get("werkzeug.server.shutdown")
            if func is None:
                return "Server shutdown not available", 500
            func()
            return "OK", 200

        @self.socketio.on("message")
        def handle_message(data):
            # Send the timestamp back for RTT calculation (expected RTT on 5 GHz Wi-Fi
            # is 7 ms) -- this is how the operator confirms the link is healthy.
            emit("echo", data["timestamp"])

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
        self.run()
        # Start the Flask server in a separate thread
        self.server_thread = threading.Thread(
            target=lambda: self.socketio.run(
                self.app,
                host="0.0.0.0",
                port=self.port,
                allow_unsafe_werkzeug=True,
                use_reloader=False,
            ),
            daemon=True,
        )
        self.server_thread.start()

    def run(self):
        # Get IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("8.8.8.8", 1))
            self.address = s.getsockname()[0]
        except Exception:
            self.address = "127.0.0.1"
        finally:
            s.close()
        print(f"Starting server at {self.address}:{self.port}")

    def stop(self):
        # Request Werkzeug shutdown endpoint
        try:
            urlopen(Request(f"http://127.0.0.1:{self.port}/__shutdown__"), timeout=1)
        except Exception:
            pass
        # Give the server a moment to stop
        time.sleep(0.2)
        # Join thread briefly; it's daemon so process can exit regardless
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=1.0)
