"""
Flask web dashboard — slot activity table, summary cards, and SSE event stream.

All routes are read-only against SQLite via get_read_connection(), which uses
WAL immutable mode and never blocks the capture writer.

Routes:
  GET /          → dashboard HTML (full page)
  GET /api/slots → JSON array of slot_stats rows, sorted by hit_count desc
  GET /api/stats → JSON summary card data (totals, active slot count, top slot)
  GET /stream    → SSE stream; pushes a 'refresh' event whenever the capture
                   process signals a new burst via the shared queue
"""

from __future__ import annotations

import json
import queue
import time
from typing import Generator

from flask import Flask, Response, jsonify, render_template, request

from meshscan import config
from meshscan.db.database import get_read_connection

app = Flask(__name__)

# Reference to the live BurstDetector — set by register_detector() in main.py
# before Flask starts receiving requests.  Using Any to avoid importing the
# detector module here and adding a cross-layer dependency.
from typing import Any as _Any
_detector: _Any = None


def register_detector(det: _Any) -> None:
    """Called by main.py after creating the BurstDetector instance."""
    global _detector
    _detector = det


# The capture loop puts a sentinel into this queue after each burst write.
# The SSE generator blocks on it and pushes a 'refresh' event to the browser.
# Maxsize prevents unbounded growth if no browser is connected.
_burst_queue: queue.Queue[None] = queue.Queue(maxsize=64)


def notify_new_burst() -> None:
    """Called by the capture loop after each DB write.  Non-blocking."""
    try:
        _burst_queue.put_nowait(None)
    except queue.Full:
        pass  # browser is keeping up or not connected — drop silently


# ---------------------------------------------------------------------------
# HTML entry point
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        poll_ms=config.UI_POLL_INTERVAL_MS,
        active_window_s=config.ACTIVE_SLOT_WINDOW_SECONDS,
    )


# ---------------------------------------------------------------------------
# JSON data endpoints (used by SSE-driven JS and polling fallback)
# ---------------------------------------------------------------------------

@app.route("/api/slots")
def api_slots() -> Response:
    """
    Return all slot_stats rows ordered by hit_count descending.
    Adds an 'is_active' boolean based on ACTIVE_SLOT_WINDOW_SECONDS.
    """
    now = time.time()
    cutoff = now - config.ACTIVE_SLOT_WINDOW_SECONDS

    conn = get_read_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM slot_stats ORDER BY hit_count DESC"
        ).fetchall()
    finally:
        conn.close()

    data = [
        {
            "slot":         r["slot"],
            "center_mhz":  r["center_mhz"],
            "bw_khz":      r["bw_khz"],
            "label":        r["label"],
            "hit_count":   r["hit_count"],
            "last_seen":   r["last_seen"],
            "peak_power_db": r["peak_power_db"],
            "is_active":   r["last_seen"] >= cutoff,
        }
        for r in rows
    ]
    return jsonify(data)


@app.route("/api/recommendation")
def api_recommendation() -> Response:
    """
    Analyze detected slot activity and return join recommendations.

    For each detected slot returns: slot number (1-indexed), frequency, BW,
    hit count, last-seen timestamp, whether it's currently active, whether it's
    the Meshtastic default slot, and a human-readable join instruction.
    """
    now = time.time()
    cutoff = now - config.ACTIVE_SLOT_WINDOW_SECONDS

    conn = get_read_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM slot_stats ORDER BY hit_count DESC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return jsonify({"mesh_detected": False, "networks": []})

    networks = []
    for r in rows:
        slot      = r["slot"]          # 0-indexed internal
        bw        = r["bw_khz"]
        is_default = slot == config.DEFAULT_SLOT_INDEX
        is_active  = r["last_seen"] >= cutoff

        if is_default:
            if bw >= 250:
                preset = "LongFast"
                join   = "Region: US  ·  Preset: LongFast  ·  Channel: Default (public)"
            else:
                preset = "LongSlow"
                join   = "Region: US  ·  Preset: LongSlow  ·  Channel: Default (public)"
        else:
            preset = f"{bw}k BW, non-default"
            join   = (
                f"Non-default sub-network on Slot {slot + 1} "
                f"({r['center_mhz']:.3f} MHz) — channel name & key required to join"
            )

        networks.append({
            "slot":        slot + 1,   # 1-indexed for display
            "center_mhz":  r["center_mhz"],
            "bw_khz":      bw,
            "hit_count":   r["hit_count"],
            "last_seen":   r["last_seen"],
            "is_active":   is_active,
            "is_default":  is_default,
            "preset":      preset,
            "join":        join,
        })

    return jsonify({"mesh_detected": True, "networks": networks})


@app.route("/api/stats")
def api_stats() -> Response:
    """
    Return summary card data:
      total_bursts   — all-time burst count for this session
      active_slots   — slots seen within ACTIVE_SLOT_WINDOW_SECONDS
      top_slot       — slot with highest hit_count (or null if no data)
      top_slot_hits  — hit count of top slot
    """
    now = time.time()
    cutoff = now - config.ACTIVE_SLOT_WINDOW_SECONDS

    conn = get_read_connection()
    try:
        total = conn.execute("SELECT SUM(hit_count) FROM slot_stats").fetchone()[0] or 0
        active = conn.execute(
            "SELECT COUNT(*) FROM slot_stats WHERE last_seen >= ?", (cutoff,)
        ).fetchone()[0]
        top_row = conn.execute(
            "SELECT slot, hit_count FROM slot_stats ORDER BY hit_count DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    return jsonify({
        "total_bursts":  total,
        "active_slots":  active,
        "top_slot":      top_row["slot"] if top_row else None,
        "top_slot_hits": top_row["hit_count"] if top_row else 0,
    })


# ---------------------------------------------------------------------------
# Detector status + live controls
# ---------------------------------------------------------------------------

@app.route("/api/detector")
def api_detector() -> Response:
    """
    Return current BurstDetector internals: noise floor, threshold, candidate
    count, and rolling 30-second window counters.  Returns {available: false}
    if the detector hasn't been registered yet (race at startup).
    """
    if _detector is None:
        return jsonify({"available": False})
    stats = _detector.get_stats()
    stats["available"] = True
    return jsonify(stats)


@app.route("/api/detector/threshold", methods=["POST"])
def api_set_threshold() -> Response:
    """
    Update ENERGY_THRESHOLD_DB live.  The capture loop reads the config value
    fresh each STFT frame, so the change takes effect within ~32 µs.

    Body: {"threshold_db": <float>}
    Clamped to [5, 40] dB.
    """
    data = request.get_json(silent=True) or {}
    try:
        val = float(data["threshold_db"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "threshold_db required"}), 400

    val = max(5.0, min(40.0, val))
    config.ENERGY_THRESHOLD_DB = val
    return jsonify({"threshold_db": val})


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------

@app.route("/stream")
def stream() -> Response:
    """
    Server-Sent Events endpoint.  The browser connects once; we push a
    'refresh' event each time _burst_queue receives a notification from the
    capture loop.  The browser re-fetches /api/slots and /api/stats on receipt.

    We also send a keepalive comment every UI_POLL_INTERVAL_MS milliseconds so
    mobile browsers (Safari) don't time out the connection.
    """
    return Response(
        _sse_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )


def _sse_generator() -> Generator[str, None, None]:
    keepalive_interval = config.UI_POLL_INTERVAL_MS / 1000.0
    while True:
        try:
            _burst_queue.get(timeout=keepalive_interval)
            yield "event: refresh\ndata: {}\n\n"
        except queue.Empty:
            # Keepalive — SSE comment lines keep the connection alive without
            # triggering an event handler on the client side.
            yield ": keepalive\n\n"


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Threaded mode required so SSE connections don't block other requests.
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
    )
