"""
Database migrations for 5 critical safeguards.
Must run BEFORE any orders execute.

Tables:
1. processed_events - Idempotency (safeguard #2)
2. order_intents - Intent persistence (safeguard #3)
3. hedge_tracking - Recursive hedge monitoring (safeguard #4)

All functions are synchronous (sqlite3 is blocking). They are wrapped
in run_in_executor() when called from async context.
"""

import sqlite3
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def create_idempotency_table(db_path: str):
    """
    Create processed_events table for idempotency.
    Tracks event_id to prevent duplicate processing.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_events (
            event_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            event_timestamp REAL NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            order_state TEXT,
            quantity REAL,
            price REAL
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_processed_events_group_id
        ON processed_events(group_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_processed_events_timestamp
        ON processed_events(processed_at)
    """)

    conn.commit()
    conn.close()
    logger.info("Created processed_events table for idempotency")


def create_order_intents_table(db_path: str):
    """
    Create order_intents table for crash safety.
    Save ORDER INTENT before sending to broker.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_intents (
            intent_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            leg_index INT NOT NULL,
            symbol TEXT NOT NULL,
            target_quantity REAL NOT NULL,
            order_type TEXT NOT NULL,
            limit_price REAL,
            intent_state TEXT DEFAULT 'INTENT_SAVED',
            broker_order_id TEXT UNIQUE,
            filled_quantity REAL DEFAULT 0,
            average_price REAL,
            intent_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_to_broker_at TIMESTAMP,
            broker_ack_at TIMESTAMP,
            completed_at TIMESTAMP,
            retry_count INT DEFAULT 0,
            last_error TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_order_intents_group_id
        ON order_intents(group_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_order_intents_broker_order_id
        ON order_intents(broker_order_id)
    """)

    conn.commit()
    conn.close()
    logger.info("Created order_intents table for crash safety")


def create_hedge_tracking_table(db_path: str):
    """
    Create hedge_tracking table for recursive hedge monitoring.
    Track each hedge placement with recursion level.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hedge_tracking (
            hedge_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            leg_index INT NOT NULL,
            symbol TEXT NOT NULL,
            hedge_target_qty REAL NOT NULL,
            hedge_type TEXT,
            recursion_level INT DEFAULT 0,
            parent_hedge_id TEXT,
            hedge_state TEXT DEFAULT 'PLACED',
            filled_quantity REAL DEFAULT 0,
            average_fill_price REAL,
            unfilled_quantity REAL,
            broker_order_id TEXT,
            placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fill_deadline TIMESTAMP,
            filled_at TIMESTAMP,
            escalated_to_trader BOOLEAN DEFAULT 0,
            trader_notes TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hedge_tracking_group_id
        ON hedge_tracking(group_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hedge_tracking_recursion_level
        ON hedge_tracking(recursion_level, hedge_state)
    """)

    conn.commit()
    conn.close()
    logger.info("Created hedge_tracking table for recursive monitoring")


def cleanup_old_events(db_path: str, retention_days: int = 1):
    """
    Clean up old processed events (older than retention_days).
    Default: 24 hour retention.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cutoff_time = datetime.now() - timedelta(days=retention_days)

    cursor.execute("""
        DELETE FROM processed_events
        WHERE processed_at < ?
    """, (cutoff_time,))

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    if deleted and deleted > 0:
        logger.info(f"Cleaned up {deleted} old processed events")


def run_all_migrations(db_path: str):
    """
    Run all 3 critical table migrations.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    """
    logger.info("=" * 60)
    logger.info("Running database migrations for 5 safeguards...")
    logger.info("=" * 60)

    try:
        create_idempotency_table(db_path)
        create_order_intents_table(db_path)
        create_hedge_tracking_table(db_path)

        logger.info("=" * 60)
        logger.info("All migrations completed successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python migrations.py <db_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    run_all_migrations(db_path)
