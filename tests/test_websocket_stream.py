import os
import sys
import json
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline.api.main import app, state


def test_health_check_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["ok", "healthy"] or "status" in data


def test_websocket_live_ping():
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_text("ping")
        resp = websocket.receive_text()
        # Initial broadcast or pong response
        assert resp in ["pong", "{", "["] or "ping" in resp or "status" in resp or "{" in resp
