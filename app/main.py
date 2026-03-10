"""FastAPI application for Enterprise Intelligence Agent."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import process_message
from app.config import get_settings
from app.database import load_sample_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum allowed message length. Aligns with typical LLM context limits and prevents
# abuse from extremely large payloads. Overlong messages are rejected (not truncated).
MAX_MESSAGE_LENGTH = 4096


async def request_id_middleware(request: Request, call_next):
    """
    Set request correlation ID for distributed tracing.
    Reads X-Request-ID from incoming request if present; otherwise generates a new UUID.
    Stores the ID in request.state for use in handlers and other middleware.
    Adds X-Request-ID to all response headers so clients can correlate logs.
    Must run before logging middleware so request_id is available in logs.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def log_requests(request: Request, call_next):
    """Log each request with method, path, status, latency, and request_id for traceability."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "%s %s %d %.2fms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id or "-",
    )
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load sample data into database."""
    settings = get_settings()
    # Use project root so paths work regardless of working directory (e.g. uvicorn from repo root)
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "sample_data.csv"
    if sample_path.exists():
        load_sample_data(str(sample_path))
    else:
        logger.warning("Sample data not found at %s", sample_path)
    yield


app = FastAPI(
    title="Enterprise Intelligence Agent",
    description="AI chatbot for enterprise analytics with SQL and ML tools",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Last-added middleware runs outermost. Add request_id first so it is set before log_requests
# executes; otherwise request_id would be missing from log lines.
app.middleware("http")(log_requests)
app.middleware("http")(request_id_middleware)


class ChatRequest(BaseModel):
    """
    Request body for chat endpoint.
    Message length is capped to avoid abuse and align with LLM context limits.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User message",
    )


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    response: str = Field(..., description="Agent response text")
    tool_calls: list[str] = Field(default_factory=list, description="Tools invoked")
    metadata: dict = Field(default_factory=dict, description="Structured metadata")


@app.get("/health")
async def health():
    """Liveness check: confirms API is up and reports mock_llm/key status for debugging."""
    settings = get_settings()
    return {
        "status": "ok",
        "mock_llm": settings.should_use_mock_llm,
        "has_openai_key": settings.has_openai_key,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user message via agent; tool calls and LLM response are handled inside process_message."""
    try:
        result = process_message(request.message)
        return ChatResponse(
            response=result["response"],
            tool_calls=result.get("tool_calls", []),
            metadata=result.get("metadata", {}),
        )
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))
