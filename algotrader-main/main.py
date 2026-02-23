from __future__ import annotations

import os
import sqlite3

import uvicorn

from src.utils.config import get_settings
from src.utils.logger import setup_logging


def _check_orphaned_execution_state() -> None:
    """Pre-startup check: warn if orphaned execution intents exist from a previous crash."""
    recovery_db = "data/execution_state.db"
    if not os.path.exists(recovery_db):
        return
    try:
        conn = sqlite3.connect(recovery_db)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM execution_intents WHERE status = 'PENDING'"
        )
        pending = cursor.fetchone()[0]
        conn.close()
        if pending > 0:
            print(
                f"\n⚠️  WARNING: {pending} orphaned execution intent(s) from previous session.\n"
                f"   Run POST /api/preflight to inspect, or DELETE data/execution_state.db to clear.\n"
            )
    except Exception:
        pass  # Table may not exist yet


def main() -> None:
    setup_logging()
    _check_orphaned_execution_state()

    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"

    uvicorn.run(
        "src.api.webapp:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
