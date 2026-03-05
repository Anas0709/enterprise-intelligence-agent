"""Database connection and SQL execution for Enterprise Intelligence Agent."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings

logger = logging.getLogger(__name__)

# Blocked SQL keywords for read-only enforcement
BLOCKED_KEYWORDS = {
    "drop", "delete", "update", "insert", "alter", "truncate",
    "create", "replace", "grant", "revoke", "execute", "exec",
}

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Get or create SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url
        # Ensure SQLite directory exists
        if db_url.startswith("sqlite"):
            path = db_url.replace("sqlite:///", "")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(db_url, echo=False)
    return _engine


def validate_sql_query(query: str) -> tuple[bool, str]:
    """
    Validate that the query is read-only (SELECT only).
    Returns (is_valid, error_message).
    """
    query_upper = query.strip().upper()
    # Must start with SELECT
    if not query_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed."

    for keyword in BLOCKED_KEYWORDS:
        # Check for keyword as whole word
        if f" {keyword.upper()} " in f" {query_upper} ":
            return False, f"Blocked keyword: {keyword.upper()}"

    return True, ""


def run_sql_query(query: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Execute a read-only SQL query and return JSON results.
    Supports parameterized queries via params dict for safe user input.
    """
    is_valid, err = validate_sql_query(query)
    if not is_valid:
        logger.warning("SQL validation failed: %s", err)
        return {"error": err, "results": []}

    try:
        engine = get_engine()
        with engine.connect() as conn:
            stmt = text(query)
            result = conn.execute(stmt, params or {})
            rows = result.fetchall()
            columns = list(result.keys())

        data = [dict(zip(columns, row)) for row in rows]
        # Convert non-JSON-serializable types
        for row in data:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
                elif hasattr(v, "item"):
                    row[k] = v.item()

        return {"results": data, "row_count": len(data)}

    except Exception as e:
        logger.exception("SQL execution error")
        return {"error": str(e), "results": []}


def load_sample_data(csv_path: str, table_name: str = "customers") -> None:
    """
    Load sample CSV into SQLite database.
    Creates table if it doesn't exist.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Sample data file not found: %s", csv_path)
        return

    df = pd.read_csv(path)
    engine = get_engine()

    # Infer SQLite types
    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
    )
    logger.info("Loaded %d rows into %s from %s", len(df), table_name, csv_path)
