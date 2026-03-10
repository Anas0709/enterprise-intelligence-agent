"""Unit tests for SQL query executor tool."""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test DB before importing app
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test_enterprise.db")

import pytest
from app.database import (
    get_business_summary,
    get_engine,
    load_sample_data,
    run_sql_query,
    validate_sql_query,
)


@pytest.fixture(scope="module")
def db_with_data():
    """Load sample data for tests."""
    sample_path = PROJECT_ROOT / "data" / "sample_data.csv"
    if sample_path.exists():
        load_sample_data(str(sample_path))
    yield
    # Cleanup: remove test db if desired
    # Path("data/test_enterprise.db").unlink(missing_ok=True)


class TestValidateSqlQuery:
    """Tests for SQL query validation."""

    def test_select_allowed(self):
        valid, err = validate_sql_query("SELECT * FROM customers")
        assert valid is True
        assert err == ""

    def test_select_with_where(self):
        valid, err = validate_sql_query("SELECT customer_id FROM customers WHERE age > 30")
        assert valid is True

    def test_drop_blocked(self):
        valid, err = validate_sql_query("DROP TABLE customers")
        assert valid is False
        assert len(err) > 0  # Any error message (SELECT-only or blocked keyword)

    def test_delete_blocked(self):
        valid, err = validate_sql_query("DELETE FROM customers WHERE id=1")
        assert valid is False

    def test_update_blocked(self):
        valid, err = validate_sql_query("UPDATE customers SET age=25")
        assert valid is False

    def test_insert_blocked(self):
        valid, err = validate_sql_query("INSERT INTO customers VALUES (1,2,3)")
        assert valid is False

    def test_select_only(self):
        valid, err = validate_sql_query("SELECT 1")
        assert valid is True


class TestRunSqlQuery:
    """Tests for SQL execution."""

    def test_invalid_query_returns_error(self):
        result = run_sql_query("DROP TABLE customers")
        assert "error" in result
        assert result.get("results") == []

    def test_select_returns_results(self, db_with_data):
        result = run_sql_query("SELECT customer_id, age, region FROM customers LIMIT 3")
        assert "error" not in result
        assert "results" in result
        assert len(result["results"]) <= 3
        if result["results"]:
            row = result["results"][0]
            assert "customer_id" in row or "age" in row or "region" in row

    def test_aggregation_works(self, db_with_data):
        result = run_sql_query("SELECT COUNT(*) as cnt FROM customers")
        assert "results" in result
        assert len(result["results"]) == 1
        assert "cnt" in result["results"][0]

    def test_parameterized_query_works(self, db_with_data):
        """Parameterized queries prevent SQL injection."""
        result = run_sql_query(
            "SELECT customer_id, age FROM customers WHERE customer_id = :cid LIMIT 1",
            params={"cid": 1},
        )
        assert "error" not in result
        assert "results" in result
        if result["results"]:
            assert result["results"][0]["customer_id"] == 1


class TestGetBusinessSummary:
    """Tests for get_business_summary database function."""

    def test_returns_expected_keys(self, db_with_data):
        """Result must contain customer_count, total_revenue, churn_rate, revenue_by_region."""
        result = get_business_summary()
        assert "customer_count" in result
        assert "total_revenue" in result
        assert "churn_rate" in result
        assert "revenue_by_region" in result

    def test_revenue_by_region_structure(self, db_with_data):
        """revenue_by_region must be a list of objects with region and revenue."""
        result = get_business_summary()
        revenue_by_region = result["revenue_by_region"]
        assert isinstance(revenue_by_region, list)
        for item in revenue_by_region:
            assert isinstance(item, dict)
            assert "region" in item
            assert "revenue" in item

    def test_revenue_by_region_ordered_descending(self, db_with_data):
        """revenue_by_region must be ordered by revenue descending."""
        result = get_business_summary()
        revenue_by_region = result["revenue_by_region"]
        revenues = [r["revenue"] for r in revenue_by_region]
        assert revenues == sorted(revenues, reverse=True)

    def test_kpis_have_sensible_types(self, db_with_data):
        """customer_count is int, total_revenue and churn_rate are numeric."""
        result = get_business_summary()
        assert isinstance(result["customer_count"], int)
        assert isinstance(result["total_revenue"], (int, float))
        assert isinstance(result["churn_rate"], (int, float))
