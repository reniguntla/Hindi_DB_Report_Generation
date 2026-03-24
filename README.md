# Hindi DB Report Generation (Streamlit + Ollama + Phi-3 + PostgreSQL)

This app lets users ask questions in **Hindi**, translates them to English, generates safe SQL with a local **Phi-3 model via Ollama**, runs the query on PostgreSQL in **read-only** mode, and explains results back in **Hindi**.

## Features

- Hindi natural language input
- Hindi → English translation using Phi-3 (Ollama)
- Schema-aware SQL generation using:
  - table/column metadata
  - data types
  - PK/FK relationships
  - table and column comments/descriptions
- SQL safety checks (blocks destructive statements)
- Read-only DB transaction execution
- Result table + downloadable CSV
- Hindi explanation of query output
- Optional generated SQL display
- Query history in UI

## Prerequisites

1. Python 3.10+
2. Local PostgreSQL instance
3. Ollama running locally
4. Phi-3 model pulled in Ollama

```bash
ollama pull phi3:mini
ollama serve
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set DB and model config via environment variables or Streamlit secrets.

### Environment variables

```bash
export PGHOST=localhost
export PGPORT=5432
export PGUSER=postgres
export PGPASSWORD=postgres
export PGDATABASE=mydb
export PGSSLMODE=prefer

export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=phi3:mini
```

### Optional: `.streamlit/secrets.toml`

```toml
[postgres]
host = "localhost"
port = 5432
user = "postgres"
password = "postgres"
dbname = "mydb"
sslmode = "prefer"
```

## Run

```bash
streamlit run app.py
```

## Safety Notes

- The app blocks SQL containing potentially destructive keywords.
- Only `SELECT` or `WITH` statements are allowed.
- Query execution uses `SET TRANSACTION READ ONLY`.

## Extensibility

You can swap models by changing `OLLAMA_MODEL` without changing business logic.
