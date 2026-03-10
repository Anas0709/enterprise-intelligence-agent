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
        # SQLite requires the parent directory to exist; create it to avoid startup failures
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
        # Pad with spaces so we match whole words (e.g. block DROP but not "droplet")
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
        # datetime/numpy types are not JSON-serializable; convert before returning
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


def get_business_summary() -> dict[str, Any]:
    """
    Fetch predefined KPIs from the customers table for high-level business overviews.
    Returns customer_count, total_revenue, churn_rate, and revenue_by_region.
    Uses fixed read-only SQL; no user input is interpolated.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            count_result = conn.execute(
                text(
                    "SELECT COUNT(*) as customer_count, COALESCE(SUM(total_spend), 0) as total_revenue "
                    "FROM customers"
                )
            )
            count_row = count_result.fetchone()
            customer_count = count_row[0] if count_row else 0
            total_revenue = float(count_row[1]) if count_row and count_row[1] is not None else 0.0

            # NULLIF avoids division by zero when table is empty
            churn_result = conn.execute(
                text(
                    "SELECT CAST(SUM(churn) AS FLOAT) / NULLIF(COUNT(*), 0) as churn_rate FROM customers"
                )
            )
            churn_row = churn_result.fetchone()
            churn_rate = float(churn_row[0]) if churn_row and churn_row[0] is not None else 0.0

            region_result = conn.execute(
                text(
                    "SELECT region, COALESCE(SUM(total_spend), 0) as revenue "
                    "FROM customers GROUP BY region ORDER BY revenue DESC"
                )
            )
            revenue_by_region = [
                {"region": row[0], "revenue": float(row[1]) if row[1] is not None else 0.0}
                for row in region_result.fetchall()
            ]

            return {
                "customer_count": customer_count,
                "total_revenue": round(total_revenue, 2),
                "churn_rate": round(churn_rate, 4),
                "revenue_by_region": revenue_by_region,
            }
    except Exception as e:
        logger.exception("get_business_summary failed: %s", e)
        return {
            "error": str(e),
            "customer_count": 0,
            "total_revenue": 0.0,
            "churn_rate": 0.0,
            "revenue_by_region": [],
        }


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
    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
    )
    logger.info("Loaded %d rows into %s from %s", len(df), table_name, csv_path)
