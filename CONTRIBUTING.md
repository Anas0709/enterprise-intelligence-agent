# Contributing to Enterprise Intelligence Agent

Thanks for your interest in contributing. This document outlines how to get set up and propose changes.

## Development Setup

```bash
git clone https://github.com/Anas0709/enterprise-intelligence-agent.git
cd enterprise-intelligence-agent
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.example .env
python train_model.py
```

## Running Tests

```bash
pytest tests/ -v
```

Tests use a separate SQLite database (`data/test_enterprise.db`). Set `DATABASE_URL` if you need a custom path.

## Code Style

- Use type hints for function signatures
- Follow existing patterns for tools, config, and API design
- Keep SQL read-only—parameterized queries for any user-provided values
- Add tests for new tools or significant changes

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) when possible:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding or updating tests
- `chore:` maintenance (deps, config, etc.)

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes with clear commits
3. Ensure tests pass (`pytest tests/ -v`)
4. Open a PR with a description of what changed and why
5. Address any review feedback

## Adding a New Tool

1. Add the tool function in `app/tools.py`
2. Add the schema to `TOOL_DEFINITIONS`
3. Register in `get_tool_executor` and `execute_tool`
4. Update the agent system prompt in `app/agent.py` if needed
5. Add tests in `tests/`
