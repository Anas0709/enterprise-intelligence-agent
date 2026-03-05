"""FastAPI application for Enterprise Intelligence Agent."""

import logging
import time
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


async def log_requests(request: Request, call_next):
    """Log incoming requests and response times."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %d %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load sample data into database."""
    settings = get_settings()
    # Default path: data/sample_data.csv relative to project root
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "sample_data.csv"
    if sample_path.exists():
        load_sample_data(str(sample_path))
    else:
        logger.warning("Sample data not found at %s", sample_path)
    yield
    # Shutdown (if needed)
    pass


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
app.middleware("http")(log_requests)


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    response: str = Field(..., description="Agent response text")
    tool_calls: list[str] = Field(default_factory=list, description="Tools invoked")
    metadata: dict = Field(default_factory=dict, description="Structured metadata")


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "ok",
        "mock_llm": settings.should_use_mock_llm,
        "has_openai_key": settings.has_openai_key,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint. Sends message to agent, executes tools if needed, returns response.
    """
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
