"""
Lightweight SQLite storage layer.
Matches the architecture diagram's "Storage" box:
customer_visits, reviews, chat_logs.

Using SQLite (a single file, storage/app.db) instead of a full DB server keeps
this deployable on free-tier hosts (Render/Railway) with zero extra setup.
"""
import sqlite3
import os
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "app.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                confidence REAL,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                sentiment TEXT,
                confidence REAL,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message TEXT,
                reply TEXT,
                intent TEXT,
                source TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                confidence REAL,
                timestamp TEXT
            )
        """)


def log_visit(customer_id: str, confidence: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO customer_visits (customer_id, confidence, timestamp) VALUES (?, ?, ?)",
            (customer_id, confidence, datetime.now(timezone.utc).isoformat()),
        )


def log_review(text: str, sentiment: str, confidence: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO reviews (text, sentiment, confidence, timestamp) VALUES (?, ?, ?, ?)",
            (text, sentiment, confidence, datetime.now(timezone.utc).isoformat()),
        )


def log_chat(session_id: str, message: str, reply: str, intent: str, source: str):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO chat_logs (session_id, message, reply, intent, source, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, message, reply, intent, source, datetime.now(timezone.utc).isoformat()),
        )


def log_product_classification(category: str, confidence: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO product_classifications (category, confidence, timestamp) VALUES (?, ?, ?)",
            (category, confidence, datetime.now(timezone.utc).isoformat()),
        )


def get_stats():
    with get_conn() as conn:
        total_visits = conn.execute("SELECT COUNT(*) c FROM customer_visits").fetchone()["c"]
        unique_customers = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) c FROM customer_visits"
        ).fetchone()["c"]
        chat_count = conn.execute("SELECT COUNT(*) c FROM chat_logs").fetchone()["c"]
        product_count = conn.execute("SELECT COUNT(*) c FROM product_classifications").fetchone()["c"]

        sentiment_rows = conn.execute(
            "SELECT sentiment, COUNT(*) c FROM reviews GROUP BY sentiment"
        ).fetchall()
        sentiment_breakdown = {row["sentiment"]: row["c"] for row in sentiment_rows}

        return {
            "total_visits": total_visits,
            "unique_customers": unique_customers,
            "sentiment_breakdown": sentiment_breakdown,
            "chatbot_messages_handled": chat_count,
            "product_classifications": product_count,
        }
