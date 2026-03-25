import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

import pandas as pd
import psycopg2
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
DEFAULT_OLLAMA_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))


@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str
    sslmode: str = "prefer"


def get_db_config() -> DBConfig:
    """Build database config from Streamlit secrets first, then env vars."""
    secrets = st.secrets.get("postgres", {}) if hasattr(st, "secrets") else {}
    return DBConfig(
        host=secrets.get("host", os.getenv("PGHOST", "localhost")),
        port=int(secrets.get("port", os.getenv("PGPORT", 5432))),
        user=secrets.get("user", os.getenv("PGUSER", "postgres")),
        password=secrets.get("password", os.getenv("PGPASSWORD", "")),
        dbname=secrets.get("dbname", os.getenv("PGDATABASE", "postgres")),
        sslmode=secrets.get("sslmode", os.getenv("PGSSLMODE", "prefer")),
    )


def get_connection(config: DBConfig):
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        dbname=config.dbname,
        sslmode=config.sslmode,
    )


@st.cache_data(ttl=300)
def fetch_schema_markdown(config: DBConfig) -> str:
    """Collect table, column, PK/FK metadata and convert to compact markdown context."""
    table_sql = """
    SELECT
      t.table_schema,
      t.table_name,
      COALESCE(obj_description((quote_ident(t.table_schema)||'.'||quote_ident(t.table_name))::regclass, 'pg_class'), '') AS table_description
    FROM information_schema.tables t
    WHERE t.table_type='BASE TABLE'
      AND t.table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY t.table_schema, t.table_name;
    """

    column_sql = """
    SELECT
      c.table_schema,
      c.table_name,
      c.column_name,
      c.data_type,
      c.is_nullable,
      c.column_default,
      COALESCE(col_description((quote_ident(c.table_schema)||'.'||quote_ident(c.table_name))::regclass,
      c.ordinal_position), '') AS column_description
    FROM information_schema.columns c
    WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY c.table_schema, c.table_name, c.ordinal_position;
    """

    key_sql = """
    SELECT
      tc.table_schema,
      tc.table_name,
      tc.constraint_name,
      tc.constraint_type,
      kcu.column_name,
      ccu.table_schema AS foreign_table_schema,
      ccu.table_name AS foreign_table_name,
      ccu.column_name AS foreign_column_name
    FROM information_schema.table_constraints tc
    LEFT JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
      AND tc.table_name = kcu.table_name
    LEFT JOIN information_schema.constraint_column_usage ccu
      ON tc.constraint_name = ccu.constraint_name
      AND tc.table_schema = ccu.table_schema
    WHERE tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
      AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY tc.table_schema, tc.table_name, tc.constraint_type;
    """

    uri = (
        f"postgresql+psycopg2://{quote_plus(config.user)}:{quote_plus(config.password)}"
        f"@{config.host}:{config.port}/{config.dbname}"
    )
    engine = create_engine(uri, pool_pre_ping=True)

    with engine.connect() as conn:
        tables = pd.read_sql_query(table_sql, conn)
        columns = pd.read_sql_query(column_sql, conn)
        keys = pd.read_sql_query(key_sql, conn)

    engine.dispose()

    lines: List[str] = ["# Database Schema Context"]
    for _, table in tables.iterrows():
        schema = table["table_schema"]
        name = table["table_name"]
        desc = table["table_description"] or "No description"
        lines.append(f"## {schema}.{name}")
        lines.append(f"Table description: {desc}")
        lines.append("Columns:")

        table_columns = columns[(columns["table_schema"] == schema) & (columns["table_name"] == name)]
        for _, col in table_columns.iterrows():
            col_desc = col["column_description"] or ""
            nullable = col["is_nullable"]
            lines.append(
                f"- {col['column_name']} ({col['data_type']}), nullable={nullable}, "
                f"default={col['column_default']}, description={col_desc}"
            )

        table_keys = keys[(keys["table_schema"] == schema) & (keys["table_name"] == name)]
        if not table_keys.empty:
            lines.append("Keys:")
            for _, key in table_keys.iterrows():
                if key["constraint_type"] == "PRIMARY KEY":
                    lines.append(f"- PK: {key['column_name']}")
                elif key["constraint_type"] == "FOREIGN KEY":
                    lines.append(
                        "- FK: "
                        f"{key['column_name']} -> {key['foreign_table_schema']}.{key['foreign_table_name']}.{key['foreign_column_name']}"
                    )
        lines.append("")

    return "\n".join(lines)


def call_ollama(
    model: str,
    prompt: str,
    host: str,
    temperature: float = 0.1,
    timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT,
    max_retries: int = DEFAULT_OLLAMA_RETRIES,
) -> str:
    """Call local Ollama generate API."""
    url = f"{host.rstrip('/')}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
    }
    errors: List[str] = []

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=(10, timeout_seconds))
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ReadTimeout:
            wait_seconds = min(2 * attempt, 8)
            errors.append(
                f"Attempt {attempt}/{max_retries}: model response timed out after {timeout_seconds}s."
            )
            time.sleep(wait_seconds)
        except requests.exceptions.RequestException as exc:
            errors.append(f"Attempt {attempt}/{max_retries}: {exc}")
            time.sleep(1)

    raise RuntimeError("Ollama call failed. " + " ".join(errors))


def translate_hindi_to_english(user_input_hindi: str, model: str, host: str) -> str:
    prompt = f"""
You are an expert Hindi-to-English translator for analytics and SQL reporting.
Translate the Hindi query into concise and clear English.
Rules:
- Preserve business entities and numbers exactly.
- Keep date/period intent explicit.
- If a Hindi word is already an English identifier (e.g., salary, employee_id), keep it unchanged.
- Return ONLY the English translation text (no explanation, no quotes).

Hindi:
{user_input_hindi}
""".strip()
    return call_ollama(model=model, prompt=prompt, host=host, temperature=0)


def generate_sql_query(english_query: str, schema_context: str, model: str, host: str) -> str:
    prompt = f"""
You are an expert PostgreSQL SQL generator.
Use only schema provided below.
Return ONLY ONE SQL query and nothing else.
Rules:
- Output must be read-only SELECT query.
- Do not invent tables/columns.
- Prefer explicit JOIN conditions with foreign keys.
- Limit rows to 200 unless user explicitly asks for more.

Schema:
{schema_context}

User request in English:
{english_query}
""".strip()
    raw_sql = call_ollama(model=model, prompt=prompt, host=host, temperature=0)
    cleaned = cleanup_sql(raw_sql)
    return extract_first_sql_statement(cleaned)


def clean_translation_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^translation\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip('"').strip("'").strip()
    return text


def cleanup_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```sql", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_first_sql_statement(text: str) -> str:
    """
    Extract the first SELECT/WITH SQL statement from model output, even when the
    model adds extra explanation text.
    """
    normalized = text.strip()
    if not normalized:
        return ""

    sql_start = re.search(r"\b(select|with)\b", normalized, flags=re.IGNORECASE)
    if not sql_start:
        return normalized

    candidate = normalized[sql_start.start():].strip()
    # Split on semicolon and keep the first statement when available.
    first_statement = candidate.split(";", 1)[0].strip()
    if first_statement:
        return first_statement + ";"

    return candidate


def is_safe_readonly_sql(sql: str) -> Tuple[bool, str]:
    sql = extract_first_sql_statement(sql)
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    if not normalized:
        return False, "Empty SQL generated."

    dangerous_keywords = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "truncate ",
        "create ",
        "grant ",
        "revoke ",
        "comment ",
    ]
    if any(kw in normalized for kw in dangerous_keywords):
        return False, "Potentially destructive SQL detected."

    if not (normalized.startswith("select") or normalized.startswith("with")):
        return False, "Only SELECT/CTE queries are allowed."

    if ";" in normalized[:-1]:
        return False, "Multiple statements are not allowed."

    return True, "OK"


def execute_readonly_query(config: DBConfig, sql: str) -> pd.DataFrame:
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY;")
            cur.execute(sql)
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=col_names)


def summarize_results_in_hindi(
    english_query: str,
    sql: str,
    df: pd.DataFrame,
    model: str,
    host: str,
) -> str:
    preview = df.head(20).to_markdown(index=False)
    prompt = f"""
User originally asked in Hindi, translated to English as:
{english_query}

SQL executed:
{sql}

Result preview table:
{preview}

Write a concise explanation in Hindi for a non-technical user:
- 3 to 6 bullet points
- Mention major insights and counts if visible
- Do not hallucinate values not present in table
""".strip()
    return call_ollama(model=model, prompt=prompt, host=host, temperature=0.2)


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="Hindi NL → PostgreSQL", layout="wide")
    st.title("हिंदी नैचुरल लैंग्वेज से PostgreSQL रिपोर्ट")
    st.caption("Ollama + Phi-3 + Streamlit | सुरक्षित Read-only SQL जनरेशन")

    init_session_state()

    with st.sidebar:
        st.header("Configuration")
        model_name = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
        ollama_host = st.text_input("Ollama host", value=DEFAULT_OLLAMA_HOST)
        show_sql = st.checkbox("Show generated SQL", value=True)
        show_schema_preview = st.checkbox("Show schema context preview", value=False)

    db_config = get_db_config()

    try:
        schema_context = fetch_schema_markdown(db_config)
        if show_schema_preview:
            with st.expander("Schema Context"):
                st.text(schema_context[:6000])
    except Exception as exc:
        st.error(f"Schema fetch failed: {exc}")
        st.stop()

    st.subheader("अपना प्रश्न हिंदी में लिखें")
    user_hindi_query = st.text_area(
        "उदाहरण: सबसे अधिक salary वाले top 5 employees की सूची दिखाओ।",
        height=120,
    )

    if st.button("Query चलाएँ", type="primary"):
        if not user_hindi_query.strip():
            st.warning("कृपया प्रश्न लिखें।")
            st.stop()

        with st.spinner("हिंदी → English translation..."):
            try:
                english_query = translate_hindi_to_english(user_hindi_query, model_name, ollama_host)
                english_query = clean_translation_response(english_query)
            except Exception as exc:
                st.error(f"Translation failed: {exc}")
                st.stop()

        st.markdown("### Translated Query (English)")
        st.info(english_query)

        with st.spinner("SQL generate हो रहा है..."):
            try:
                sql = generate_sql_query(english_query, schema_context, model_name, ollama_host)
            except Exception as exc:
                st.error(f"SQL generation failed: {exc}")
                st.stop()

        is_safe, reason = is_safe_readonly_sql(sql)
        if not is_safe:
            st.error(f"Unsafe SQL blocked: {reason}")
            if show_sql:
                st.code(sql, language="sql")
            st.stop()

        with st.spinner("Database query execute हो रही है..."):
            try:
                df = execute_readonly_query(db_config, sql)
            except Exception as exc:
                st.error(f"Query execution failed: {exc}")
                if show_sql:
                    st.code(sql, language="sql")
                st.stop()

        if show_sql:
            st.markdown("### Generated SQL")
            st.code(sql, language="sql")

        st.markdown("### Query Results")
        if df.empty:
            st.info("कोई परिणाम नहीं मिला।")
        else:
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "CSV डाउनलोड करें",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_results.csv",
                mime="text/csv",
            )

        with st.spinner("हिंदी में summary तैयार हो रही है..."):
            try:
                hindi_summary = summarize_results_in_hindi(english_query, sql, df, model_name, ollama_host)
            except Exception as exc:
                hindi_summary = f"Summary generation failed: {exc}"

        st.markdown("### हिंदी Explanation")
        st.write(hindi_summary)

        st.session_state.history.insert(
            0,
            {
                "hindi_query": user_hindi_query,
                "english_query": english_query,
                "sql": sql,
                "rows": len(df),
            },
        )

    if st.session_state.history:
        st.markdown("---")
        st.subheader("Query History")
        for i, item in enumerate(st.session_state.history[:10], start=1):
            with st.expander(f"{i}. {item['hindi_query'][:70]}"):
                st.write(f"English: {item['english_query']}")
                st.code(item["sql"], language="sql")
                st.write(f"Rows: {item['rows']}")


if __name__ == "__main__":
    main()
