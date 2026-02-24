"""Agent layer - LLM orchestration with tool calling."""

import json
import logging
from typing import Any, Optional

from app.config import get_settings
from app.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)


def _get_openai_client():
    """Get OpenAI client (lazy import)."""
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key or "sk-placeholder")


def _mock_llm_response(message: str) -> dict[str, Any]:
    """Return a stubbed response when no API key or mock mode."""
    return {
        "response": (
            "I'm running in mock mode. Please set OPENAI_API_KEY in your .env to enable "
            "full AI capabilities. I can help with SQL queries and churn predictions once configured."
        ),
        "tool_calls": [],
        "metadata": {
            "insight_summary": "Mock mode - no analysis performed.",
            "confidence_level": "low",
            "data_sources_used": [],
        },
    }


def _build_metadata(
    tool_calls: list[str],
    data_sources: Optional[set] = None,
) -> dict[str, Any]:
    """Build structured metadata from tool usage."""
    sources = data_sources or set()
    if "run_sql_query" in tool_calls:
        sources.add("sql")
    if "predict_churn" in tool_calls:
        sources.add("ml_model")

    confidence = "high" if sources else "medium"
    return {
        "data_sources_used": list(sources),
        "confidence_level": confidence,
    }


def _call_llm_with_tools(message: str) -> dict[str, Any]:
    """Call OpenAI API with tool/function calling."""
    settings = get_settings()
    client = _get_openai_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Enterprise Intelligence Agent. You help users analyze business data "
                "using SQL queries and ML predictions. When asked about revenue, regions, customers, "
                "or analytics, use run_sql_query. When asked about churn risk for a specific customer, "
                "use predict_churn. Always summarize results clearly and provide actionable insights. "
                "Include structured metadata: insight_summary, confidence_level, data_sources_used."
            ),
        },
        {"role": "user", "content": message},
    ]

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        tool_calls_used = []
        data_sources = set()

        # Handle tool calls
        while choice.message.tool_calls:
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                name = tc.function.name
                tool_calls_used.append(name)
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                output = execute_tool(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            # Get next LLM response
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            choice = response.choices[0]

        content = choice.message.content or ""

        # Extract or build insight summary
        insight = content[:200] + "..." if len(content) > 200 else content
        metadata = _build_metadata(tool_calls_used, data_sources)
        metadata["insight_summary"] = insight

        return {
            "response": content,
            "tool_calls": tool_calls_used,
            "metadata": metadata,
        }

    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return {
            "response": f"I encountered an error: {str(e)}. Please check your API key and try again.",
            "tool_calls": [],
            "metadata": {
                "insight_summary": str(e),
                "confidence_level": "low",
                "data_sources_used": [],
            },
        }


def process_message(message: str) -> dict[str, Any]:
    """
    Process a user message through the agent.
    Returns dict with response, tool_calls, and metadata.
    """
    settings = get_settings()

    if settings.should_use_mock_llm:
        return _mock_llm_response(message)

    return _call_llm_with_tools(message)
