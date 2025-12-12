import psycopg2
import psycopg2.extras
import streamlit as st


def _get_secrets():
    required = ["PG_HOST", "PG_DATABASE", "PG_USER", "PG_PASSWORD", "PG_PORT"]
    missing = [k for k in required if k not in st.secrets]
    if missing:
        raise RuntimeError(
            "Missing keys in secrets: " + ", ".join(missing) + ". "
            "Add them in .streamlit/secrets.toml or Streamlit Cloud secrets."
        )
    return st.secrets


def db_ok():
    try:
        _ = _get_secrets()
        return True
    except Exception:
        return False


def get_conn():
    s = _get_secrets()
    return psycopg2.connect(
        host=s["PG_HOST"],
        database=s["PG_DATABASE"],
        user=s["PG_USER"],
        password=s["PG_PASSWORD"],
        port=str(s.get("PG_PORT", "5432")),
        sslmode="require",
    )


def get_user(username):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user


def create_user(username):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "INSERT INTO users (username) VALUES (%s) RETURNING *",
        (username,),
    )
    user = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return user


def add_interaction(user_id, item_id, action="viewed"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO interactions (user_id, item_id, action) VALUES (%s, %s, %s)",
        (user_id, item_id, action),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_user_history(user_id):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT item_id FROM interactions WHERE user_id = %s ORDER BY ts DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r["item_id"] for r in rows]
