#!/usr/bin/env python3
"""
Weekly maintenance runner for enterprise-intelligence-agent.

Runs the test suite and appends a verification entry to the maintenance log.
Designed to be invoked by .github/workflows/weekly-maintenance.yml on a schedule.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / ".github" / "maintenance-log.md"


def run_tests() -> tuple[int, str]:
    """Run pytest and return (exit_code, combined_output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={
            **dict(__import__("os").environ),
            "DATABASE_URL": "sqlite:///./data/test_enterprise.db",
        },
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output.strip()


def append_log_entry(passed: bool, summary_line: str) -> None:
    """Append a dated verification entry to the maintenance log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# Maintenance Log\n\n"
            "Automated weekly verification of the test suite. "
            "Updated by `.github/workflows/weekly-maintenance.yml`.\n\n",
            encoding="utf-8",
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "passed" if passed else "failed"
    entry = f"- **{timestamp}** — tests {status}. `{summary_line}`\n"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)


def main() -> int:
    code, output = run_tests()
    summary = output.splitlines()[-1] if output else "no output"
    append_log_entry(passed=code == 0, summary_line=summary)
    if output:
        print(output)
    # Always exit 0 so the workflow still commits the log entry when tests fail
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
