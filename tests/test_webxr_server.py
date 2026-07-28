# tests/test_webxr_server.py
"""Tests for the WebXR phone-teleop Flask/Socket.IO server.

These exist at all because `WebServer.__init__` no longer binds a port or spawns a
thread: construction builds the Flask app, the Socket.IO handlers and the message
deque, and `start()` is what touches the network. That split is what makes the
handlers testable through `SocketIO.test_client` with nothing listening.
"""
import socket
import urllib.error
from collections import deque
from urllib.request import urlopen

import pytest

from momagen.utils.webxr_server import _DEFAULT_QUEUE_MAXLEN, WebServer


def _free_port():
    """Ask the OS for a port, then release it, so a later bind can claim it."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------
# Construction is side-effect free (this is what makes everything below possible)
# --------------------------------------------------------------------------

def test_constructing_does_not_bind_a_port_or_spawn_a_thread():
    # Two servers configured for the SAME port must construct without conflict:
    # binding is start()'s job, not the constructor's.
    port = _free_port()
    a = WebServer(port=port)
    b = WebServer(port=port)
    assert a.server_thread is None and b.server_thread is None
    # Nothing is listening, so nothing can be fetched.
    with pytest.raises(urllib.error.URLError):
        urlopen(f"http://127.0.0.1:{port}/", timeout=2)


def test_default_queue_is_bounded():
    # A stalled sim loop must drop stale poses, not grow memory without bound.
    assert WebServer().queue.maxlen == _DEFAULT_QUEUE_MAXLEN


# --------------------------------------------------------------------------
# Socket handlers, exercised with no port bound
# --------------------------------------------------------------------------

def test_message_handler_pushes_onto_the_queue():
    q = deque()
    server = WebServer(queue=q)
    client = server.socketio.test_client(server.app)
    client.emit("message", {"teleop_mode": "arm", "timestamp": 1234})
    assert list(q) == [{"teleop_mode": "arm", "timestamp": 1234}]
    client.disconnect()


def test_echo_reply_returns_the_timestamp_for_rtt():
    # The echo is how the operator confirms the phone link is healthy (~7 ms on
    # 5 GHz Wi-Fi); it must return the timestamp it was sent, unmodified.
    server = WebServer()
    client = server.socketio.test_client(server.app)
    client.emit("message", {"timestamp": 987654})
    received = client.get_received()
    echoes = [r for r in received if r["name"] == "echo"]
    assert echoes and echoes[0]["args"] == [987654]
    client.disconnect()


def test_message_without_timestamp_is_queued_and_does_not_raise():
    # `emit("echo", data["timestamp"])` raised KeyError on any message lacking a
    # timestamp, which killed the handler and dropped the pose on the floor.
    q = deque()
    server = WebServer(queue=q)
    client = server.socketio.test_client(server.app)
    client.emit("message", {"teleop_mode": "arm"})
    assert list(q) == [{"teleop_mode": "arm"}], "pose must still reach the sim loop"
    assert not [r for r in client.get_received() if r["name"] == "echo"]
    client.disconnect()


def test_non_dict_message_does_not_raise():
    q = deque()
    server = WebServer(queue=q)
    client = server.socketio.test_client(server.app)
    client.emit("message", "not-a-dict")
    assert list(q) == ["not-a-dict"]
    client.disconnect()


def test_save_and_discard_are_queued_only_when_recording():
    q = deque()
    off = WebServer(queue=q, record_enabled=False)
    client = off.socketio.test_client(off.app)
    client.emit("save_episode")
    client.emit("discard_episode")
    assert list(q) == []
    client.disconnect()

    q2 = deque()
    on = WebServer(queue=q2, record_enabled=True)
    client2 = on.socketio.test_client(on.app)
    client2.emit("save_episode")
    client2.emit("discard_episode")
    assert [m["state_update"] for m in q2] == ["save_episode", "discard_episode"]
    client2.disconnect()


# --------------------------------------------------------------------------
# Real network lifecycle
# --------------------------------------------------------------------------

def test_start_stop_start_cycle_on_the_same_port():
    # stop() used to poke `request.environ["werkzeug.server.shutdown"]`, removed in
    # Werkzeug 2.1 (this checkout runs 3.1.8): the endpoint 500'd, the urlopen raised,
    # a bare `except Exception` swallowed it, and the server kept serving — so the
    # NEXT WebServer died with "Address already in use". This pins the real cycle.
    first = WebServer(host="127.0.0.1", port=0)
    first.start()
    port = first.port
    assert port != 0, "start() must record the actually-bound port"
    assert urlopen(f"http://127.0.0.1:{port}/", timeout=5).status == 200
    first.stop()

    # The port must be genuinely released, not just "the thread is a daemon".
    with pytest.raises(urllib.error.URLError):
        urlopen(f"http://127.0.0.1:{port}/", timeout=2)

    second = WebServer(host="127.0.0.1", port=port)
    second.start()   # must not raise OSError: Address already in use
    try:
        assert urlopen(f"http://127.0.0.1:{port}/", timeout=5).status == 200
    finally:
        second.stop()


def test_stop_is_safe_before_start_and_twice():
    server = WebServer(host="127.0.0.1", port=0)
    server.stop()          # never started
    server.start()
    server.stop()
    server.stop()          # already stopped


def test_started_server_serves_the_vendored_client_assets():
    # template_folder and static_folder both point at momagen/assets/webxr; if that
    # wiring breaks the phone gets a 404 instead of the teleop page.
    server = WebServer(host="127.0.0.1", port=0)
    server.start()
    try:
        body = urlopen(f"http://127.0.0.1:{server.port}/", timeout=5).read().decode()
        assert "RECORD_ENABLED = false" in body, "index.html template did not render"
        assert urlopen(
            f"http://127.0.0.1:{server.port}/static/webxr-button.js", timeout=5
        ).status == 200
    finally:
        server.stop()
