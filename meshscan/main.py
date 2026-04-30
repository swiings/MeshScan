"""
MeshScan entry point.

Wires together:
  1. DB init + session open
  2. SDRplayCapture → BurstDetector → slot_mapper → Database
  3. Flask web dashboard (in a daemon thread)
  4. Clean shutdown on SIGINT / SCAN_DURATION_SECONDS elapsed

Run with:  ./run.sh   (from the MeshScan/ directory)
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from meshscan import config
from meshscan.capture.burst_detector import BurstDetector, BurstEvent
from meshscan.capture.slot_mapper import map_burst
from meshscan.capture.sdrplay_capture import SDRplayCapture
from meshscan.db.database import Database, init_db
from meshscan.web.app import app as flask_app, notify_new_burst, register_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("meshscan")


def main() -> None:
    log.info("MeshScan starting — centre %.3f MHz, %.1f Msps",
             config.SDR_CENTER_FREQ_MHZ, config.SDR_SAMPLE_RATE_MSPS)

    # --- DB setup -----------------------------------------------------------
    init_db()
    db = Database()
    session_id = db.open_session(notes="meshscan auto-session")
    log.info("Session %d opened (DB: %s)", session_id, config.DB_PATH)

    # --- Burst callback: slot-map → DB write → SSE notify -------------------
    # Fires on the SDRplayCapture processor thread; must be fast.
    def on_burst(evt: BurstEvent) -> None:
        match = map_burst(evt.center_mhz, evt.bandwidth_khz)
        db.insert_burst(
            session_id   = session_id,
            slot         = match.slot,
            center_mhz   = match.center_mhz,
            bw_khz       = match.bw_khz,
            label        = match.label,
            peak_power_db= evt.peak_power_db,
            detected_at  = evt.timestamp_utc,
        )
        notify_new_burst()
        log.debug(
            "Burst: slot %d  %.3f MHz  BW %d kHz  %.1f dB  (%d frames)  %s",
            match.slot, match.center_mhz, match.bw_khz,
            evt.peak_power_db, evt.frame_count, match.label,
        )

    # --- Flask thread (daemon — dies with the main process) -----------------
    flask_thread = threading.Thread(
        target=lambda: flask_app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        ),
        name="meshscan-flask",
        daemon=True,
    )
    flask_thread.start()
    log.info("Dashboard at http://%s:%d", config.FLASK_HOST, config.FLASK_PORT)

    # --- Shutdown coordination ----------------------------------------------
    stop_event = threading.Event()

    def _handle_sigint(sig, frame):  # noqa: ANN001
        log.info("SIGINT received — shutting down…")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    # --- SDR capture --------------------------------------------------------
    detector = BurstDetector(on_burst=on_burst)
    register_detector(detector)
    cap = SDRplayCapture(consumer=detector.push_block)

    try:
        cap.start()
        log.info("Streaming. Press Ctrl-C to stop.")

        deadline = (
            time.time() + config.SCAN_DURATION_SECONDS
            if config.SCAN_DURATION_SECONDS > 0
            else None
        )

        while not stop_event.is_set():
            if deadline is not None and time.time() >= deadline:
                log.info("SCAN_DURATION_SECONDS elapsed — stopping.")
                break
            time.sleep(0.25)

    finally:
        cap.stop()
        db.close_session(session_id)
        db.close()
        log.info("Session %d closed. Goodbye.", session_id)


if __name__ == "__main__":
    main()
