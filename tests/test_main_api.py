import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import get_settings


# Ensure API tests use a dedicated DB file and that the SQLAlchemy engine
# cache (singleton) is cleared before importing the FastAPI app.
os.environ["DATABASE_URL"] = "sqlite:///./data/test_enterprise_main.db"

import app.database as database  # noqa: E402

database._engine = None  # Reset cached engine so DATABASE_URL takes effect.

from app.main import app  # noqa: E402


client = TestClient(app)


def test_health_returns_ok():
    """Liveness endpoint should always respond when the process is running."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "mock_llm" in data
    assert "has_openai_key" in data


def test_ready_returns_200_when_dependencies_available():
    """Readiness should pass once sample data and the trained model are present."""
    model_path = Path(get_settings().model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        model_path.write_bytes(b"\x80")

    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert data["checks"]["database"] == "ok"
    assert data["checks"]["model"] == "ok"


def test_ready_returns_503_when_database_unavailable(monkeypatch):
    """Load balancers rely on 503 to stop routing traffic to unhealthy instances."""
    monkeypatch.setattr("app.main.check_db_connection", lambda: False)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"


def test_x_request_id_is_propagated():
    """The server must echo X-Request-ID so clients can correlate traces."""
    r = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"X-Request-ID": "test-trace-123"},
    )
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == "test-trace-123"


def test_x_request_id_is_generated_when_missing():
    """If the client doesn't send X-Request-ID, the server must generate one."""
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") is not None
    assert r.headers.get("X-Request-ID") != ""


def test_message_length_limit_rejects_overlong_payload():
    """Overlong messages should be rejected via standard 422 validation."""
    r = client.post("/chat", json={"message": "x" * 4097})
    assert r.status_code == 422


def test_message_length_limit_allows_max_size():
    """The limit is inclusive at the configured maximum length."""
    r = client.post("/chat", json={"message": "x" * 4096})
    assert r.status_code == 200


def test_chat_500_does_not_leak_internal_errors():
    """500 responses must return a generic message, not internal exception details."""
    with patch("app.main.process_message", side_effect=RuntimeError("internal secret")):
        r = client.post("/chat", json={"message": "trigger error"})
    assert r.status_code == 500
    data = r.json()
    assert "internal secret" not in data.get("detail", "")
    assert "An unexpected error occurred" in data.get("detail", "")

