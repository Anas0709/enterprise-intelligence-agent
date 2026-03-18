import os

from fastapi.testclient import TestClient


# Ensure API tests use a dedicated DB file and that the SQLAlchemy engine
# cache (singleton) is cleared before importing the FastAPI app.
os.environ["DATABASE_URL"] = "sqlite:///./data/test_enterprise_main.db"

import app.database as database  # noqa: E402

database._engine = None  # Reset cached engine so DATABASE_URL takes effect.

from app.main import app  # noqa: E402


client = TestClient(app)


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

