"""Tool definitions for the Enterprise Intelligence Agent."""

import json
import logging
from typing import Any, Callable, Optional

from app.database import run_sql_query
from app.ml_model import predict_churn as ml_predict_churn

logger = logging.getLogger(__name__)


def execute_sql_query(query: str) -> str:
    """
    Execute a read-only SQL query against the enterprise database.
    Only SELECT queries are allowed. Returns JSON string of results.
    """
    result = run_sql_query(query)
    return json.dumps(result, default=str)


def predict_churn(customer_id: int) -> str:
    """
    Predict churn risk for a given customer.
    Returns JSON with customer_id, churn_probability, and risk_level.
    """
    result = ml_predict_churn(customer_id)
    return json.dumps(result)


# Tool schema for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Execute a read-only SQL query against the enterprise database. Use for revenue analysis, customer statistics, regional breakdowns, churn counts, and any analytical queries. Only SELECT statements are allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute. Schema: customers table has customer_id (int), age (int), region (string), total_spend (float), churn (0/1), signup_date (date).",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn",
            "description": "Predict the churn probability and risk level for a specific customer by their customer_id. Use when the user asks about churn risk, retention prediction, or likelihood of a customer leaving.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The unique customer ID to predict churn for.",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
]


def get_tool_executor(name: str) -> Optional[Callable[..., str]]:
    """Get the executor function for a tool by name."""
    executors = {
        "run_sql_query": execute_sql_query,
        "predict_churn": predict_churn,
    }
    return executors.get(name)


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a tool by name with the given arguments.
    Returns the tool output as a string.
    """
    executor = get_tool_executor(name)
    if executor is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        if name == "run_sql_query":
            return executor(arguments.get("query", ""))
        elif name == "predict_churn":
            return executor(int(arguments.get("customer_id", 0)))
        return json.dumps({"error": "Invalid tool arguments"})
    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        return json.dumps({"error": str(e)})
