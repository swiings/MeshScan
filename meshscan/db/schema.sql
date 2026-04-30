-- MeshScan database schema
-- WAL mode and page size are set at connection time in database.py, not here.

-- One row per confirmed burst detection
CREATE TABLE IF NOT EXISTS bursts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at     REAL    NOT NULL,   -- Unix timestamp (UTC)
    slot            INTEGER NOT NULL,
    center_mhz      REAL    NOT NULL,
    bw_khz          INTEGER NOT NULL,
    label           TEXT    NOT NULL,
    peak_power_db   REAL    NOT NULL,
    session_id      INTEGER NOT NULL REFERENCES sessions(id)
);

-- Aggregate per-slot stats, upserted after each burst
CREATE TABLE IF NOT EXISTS slot_stats (
    slot            INTEGER PRIMARY KEY,
    center_mhz      REAL    NOT NULL,
    bw_khz          INTEGER NOT NULL,
    label           TEXT    NOT NULL,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    last_seen       REAL    NOT NULL,   -- Unix timestamp
    peak_power_db   REAL    NOT NULL
);

-- Session lifecycle markers so the dashboard can show session boundaries
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      REAL    NOT NULL,
    closed_at       REAL,               -- NULL = session still running
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_bursts_slot_time ON bursts(slot, detected_at);
CREATE INDEX IF NOT EXISTS idx_bursts_session   ON bursts(session_id);
