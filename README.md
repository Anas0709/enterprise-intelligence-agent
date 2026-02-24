# Enterprise Intelligence Agent

A backend-driven AI chatbot that connects to enterprise data, executes analytical SQL queries, runs ML predictions, and returns structured business insights. Built with FastAPI, OpenAI tool-calling, and a modular, pluggable architecture.

## Project Intent

This project showcases a real-world AI/ML integration where an LLM performs decision support over enterprise data and predictive models. It highlights handling SQL, ML pipelines, API design, and modular code structure—skills directly applicable to production AI systems.

## Project Overview

The Enterprise Intelligence Agent simulates an enterprise-grade AI system that:

- **Connects** to structured enterprise data (SQL database)
- **Executes** analytical SQL queries (read-only)
- **Integrates** a machine learning churn prediction model
- **Uses** LLM tool-calling to dynamically choose actions
- **Returns** structured JSON business insights

The system is extensible and open-source: plug in your own LLM API key, connect your database, replace the ML model, or add new tools.

## Architecture

```
┌─────────┐     POST /chat      ┌──────────────────┐
│  User   │ ──────────────────► │  FastAPI         │
└─────────┘                     │  (main.py)       │
                                └────────┬─────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │  Agent Layer     │
                                │  (agent.py)      │
                                │  LLM + Tools     │
                                └────────┬─────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
           │ SQL Executor │    │ ML Predictor │    │ (extensible) │
           │ (tools.py)   │    │ (ml_model.py)│    │              │
           └──────┬───────┘    └──────┬───────┘    └──────────────┘
                  │                   │
                  ▼                   ▼
           ┌──────────────┐    ┌──────────────┐
           │ SQLite/DB    │    │ model.pkl    │
           │ (database.py)│    │ (sklearn)    │
           └──────────────┘    └──────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- OpenAI API key (optional for mock mode)

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd enterprise-intelligence-agent
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Train the Model

```bash
python train_model.py
```

This creates `models/model.pkl` from `data/sample_data.csv` and prints test accuracy and AUC.

### 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

The API starts at `http://localhost:8000`. Sample data is auto-loaded into SQLite on startup.

## Example curl Requests

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat - Revenue by Region

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is total revenue by region?"}'
```

### Chat - Churn Prediction

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Predict churn risk for customer 10"}'
```

### Chat - Business Summary

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize business risks"}'
```

## Agent Tool-Calling Flow

1. **User** sends a message to `POST /chat`.
2. **Agent** receives the message and forwards it to the LLM (OpenAI) with tool definitions.
3. **LLM** decides which tool(s) to call (or none) based on the message.
4. **Tools** execute:
   - `run_sql_query`: Validates (read-only), executes SQL, returns JSON.
   - `predict_churn`: Loads model, extracts features, returns probability + risk level.
5. **Agent** sends tool results back to the LLM.
6. **LLM** generates a natural language response with insights.
7. **Response** includes structured metadata: `insight_summary`, `confidence_level`, `data_sources_used`.

## Response Format

Example response for *"What is total revenue by region?"*:

```json
{
  "response": "Based on the data, total revenue by region: North ($6,647.75), South ($6,530.25), East ($11,412.00), West ($7,961.25). North and East together account for the highest combined revenue.",
  "tool_calls": ["run_sql_query"],
  "metadata": {
    "insight_summary": "Revenue higher in North/East; West and South show room for growth.",
    "confidence_level": "high",
    "data_sources_used": ["sql"]
  }
}
```

Example response for *"Predict churn risk for customer 10"*:

```json
{
  "response": "Customer 10 has a churn probability of 0.23 (23%), which falls in the low-risk category. This customer appears stable based on their spend and region profile.",
  "tool_calls": ["predict_churn"],
  "metadata": {
    "insight_summary": "Low churn risk; no immediate retention action needed.",
    "confidence_level": "high",
    "data_sources_used": ["ml_model"]
  }
}
```

## Model Metrics

The churn prediction model is evaluated on a held-out 20% test set (sample dataset, 25 rows):

| Metric        | Value   |
|---------------|---------|
| Test Accuracy | 80.00%  |
| Test AUC-ROC  | 1.00    |

Features: `age`, `total_spend`, `region` (one-hot encoded). Risk levels: low (&lt;30% prob), medium (30–60%), high (&gt;60%).

## Database Schema (Default)

| Column       | Type   | Description              |
|-------------|--------|--------------------------|
| customer_id | int    | Unique customer ID       |
| age         | int    | Customer age             |
| region      | string | north/south/east/west    |
| total_spend | float  | Total spend amount       |
| churn       | 0/1    | Churned (1) or not (0)   |
| signup_date | date   | Signup date              |

## Extending the System

### Add a New Tool

1. Define the tool in `app/tools.py`:
   - Add a function (e.g. `my_new_tool(arg: str) -> str`)
   - Add schema to `TOOL_DEFINITIONS`
   - Register in `get_tool_executor` and `execute_tool`

2. Update the agent system prompt in `app/agent.py` if needed.

### Swap LLM Provider

- Replace OpenAI client in `app/agent.py` with another provider (Anthropic, etc.).
- Keep the same tool-calling interface (`tools` array, `execute_tool`).

### Swap Database

- Change `DATABASE_URL` in `.env` (e.g. `postgresql://user:pass@host/db`).
- Ensure SQLAlchemy supports the driver (add `psycopg2` for PostgreSQL).

## Docker

```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
```

## Limitations & Future Improvements

- **SQL**: Read-only; no parameterized queries in tool (relies on LLM to build safe queries).
- **ML**: Single model (churn); feature encoding is coupled to training script.
- **LLM**: OpenAI only (extensible but not abstracted).
- **Future**: Embeddings + vector search, logging middleware, Streamlit UI, unit tests for SQL tool.

## License

MIT
