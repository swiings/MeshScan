"""
SQLite access layer — session lifecycle, burst inserts, and slot_stats upserts.

Uses WAL mode so Flask reads during active capture don't block writes.
The Database class holds one long-lived write connection for the capture loop.
Flask routes call get_read_connection() for short-lived read-only connections;
these don't need to be explicitly closed when used as a context manager.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from meshscan import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply connection-level settings that must be set before first use."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # page_size only takes effect before the first table is created; harmless
    # to set on an existing DB but SQLite silently ignores it in that case.
    conn.execute(f"PRAGMA page_size={config.DB_PAGE_SIZE}")


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


# ---------------------------------------------------------------------------
# Schema init — safe to call on every startup
# ---------------------------------------------------------------------------

def init_db(path: str = config.DB_PATH) -> None:
    """Create tables and indexes if they don't exist."""
    schema = (Path(__file__).parent / "schema.sql").read_text()
    conn = _connect(path)
    conn.executescript(schema)
    conn.close()


# ---------------------------------------------------------------------------
# Write-side handle (one instance per process, lives in the capture loop)
# ---------------------------------------------------------------------------

class Database:
    """
    Write-side database handle.  Holds a single persistent connection so that
    every burst write doesn't pay connection-open overhead at 39 k frames/sec.

    Not thread-safe — all calls must come from the same thread (the capture
    loop that owns this instance).  Flask reads use get_read_connection().
    """

    def __init__(self, path: str = config.DB_PATH) -> None:
        self._conn = _connect(path)

    # --- Session lifecycle -------------------------------------------------

    def open_session(self, notes: Optional[str] = None) -> int:
        """Insert a new session row and return its id."""
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at, notes) VALUES (?, ?)",
            (time.time(), notes),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def close_session(self, session_id: int) -> None:
        """Stamp closed_at so the dashboard can show session boundaries."""
        self._conn.execute(
            "UPDATE sessions SET closed_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        self._conn.commit()

    # --- Burst insert + slot_stats upsert ---------------------------------

    def insert_burst(
        self,
        session_id: int,
        slot: int,
        center_mhz: float,
        bw_khz: int,
        label: str,
        peak_power_db: float,
        detected_at: Optional[float] = None,
    ) -> None:
        """
        Insert one burst row and upsert the per-slot aggregate.

        Both writes happen in a single transaction so the aggregate is always
        consistent with the raw burst log even if the process is killed mid-run.
        """
        ts = detected_at if detected_at is not None else time.time()

        with self._conn:  # auto-commits on exit, rolls back on exception
            self._conn.execute(
                """
                INSERT INTO bursts
                    (detected_at, slot, center_mhz, bw_khz, label, peak_power_db, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, slot, center_mhz, bw_khz, label, peak_power_db, session_id),
            )

            # Upsert slot_stats:
            #   - On first detection for a slot: insert the full row.
            #   - On subsequent detections: increment hit_count, update last_seen
            #     and peak_power_db if the new peak is higher.
            self._conn.execute(
                """
                INSERT INTO slot_stats
                    (slot, center_mhz, bw_khz, label, hit_count, last_seen, peak_power_db)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(slot) DO UPDATE SET
                    hit_count     = hit_count + 1,
                    bw_khz        = excluded.bw_khz,
                    label         = excluded.label,
                    last_seen     = excluded.last_seen,
                    peak_power_db = MAX(peak_power_db, excluded.peak_power_db)
                """,
                (slot, center_mhz, bw_khz, label, ts, peak_power_db),
            )

    # --- Housekeeping ------------------------------------------------------

    def close(self) -> None:
        """Flush and close the write connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Read-side (Flask route handlers call this per request)
# ---------------------------------------------------------------------------

def get_read_connection(path: Optional[str] = None) -> sqlite3.Connection:
    """
    Return a read-only connection for Flask route handlers.

    WAL mode allows unlimited concurrent readers alongside the active writer,
    so there's no need for connection pooling — a new connection per request
    is cheap and avoids any shared-state complexity.

    Callers are responsible for closing the connection.
    """
    from urllib.parse import quote
    resolved = path if path is not None else config.DB_PATH
    # mode=ro prevents accidental writes; don't use immutable=1 because that
    # disables WAL reads and the capture loop writes via WAL.
    uri = f"file:{quote(resolved)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn
