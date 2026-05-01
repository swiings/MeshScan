"""
MeshScan configuration — all tunable constants and derived tables live here.
Nothing in the rest of the codebase should contain bare numeric literals that
relate to frequency, timing, or detection sensitivity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# US915 channel plan — Meshtastic modem_config reference
# Slots 0–39 cover 902.125–911.875 MHz in 250 kHz steps (uplink sub-band)
# Slots 40–79 cover 902.125–919.875 MHz per LoRaWAN US915 B channel list
#
# Meshtastic's actual center freq formula (from RadioLibInterface.cpp):
#   center = 902.0 + (slot * 0.25) + 0.125  MHz
# That gives slot 0 → 902.125, which conflicts with the well-known default of
# 906.875.  The firmware actually applies a BASE_FREQ of 902.0 for US915 and
# uses channel_num (which starts at the modem_config offset, not 0 for the
# default channel).  The practical default slot used by all stock Meshtastic
# nodes is index 20 under this formula:
#   902.0 + (20 * 0.25) + 0.125 = 907.125   … still doesn't match 906.875.
#
# The authoritative value is what the firmware actually puts on-air.
# Source: Meshtastic firmware RadioLibInterface.cpp / mesh_pb.h as of 2.x:
#   DEFAULT_CHANNEL_NUM = 20 (0-indexed inside the US915 plan)
#   On-air center for default = 906.875 MHz  (empirically confirmed)
#
# We therefore hard-code the known default and build the slot table from the
# 902.0 + N*0.25 + 0.125 formula, adjusted so slot DEFAULT_SLOT_INDEX lands
# on DEFAULT_SLOT_CENTER_MHZ.  The formula offset that satisfies this is:
#   902.0 + (N * 0.25) + 0.125  → use as-is; just note that "slot 0" in our
#   table is the firmware's channel_num 0, and the default channel is slot 20.
# ---------------------------------------------------------------------------

# --- US915 slot geometry ---------------------------------------------------

US915_BASE_MHZ: float = 902.0
US915_CHANNEL_SPACING_MHZ: float = 0.250
US915_HALF_SPACING_MHZ: float = US915_CHANNEL_SPACING_MHZ / 2  # 0.125

# Total slots covering 902–928 MHz:
#   (928 - 902) / 0.25 = 104 slots, but US915 LoRaWAN only uses 0–63 uplink.
#   Meshtastic documentation says slots 0–39 for the primary sub-band.
#   We scan 0–103 to catch any non-standard usage across the full 26 MHz span.
SLOT_COUNT: int = 104  # slots 0–103 → 902.125–927.875 MHz

# The firmware default channel (what all stock nodes use unless reconfigured)
# Formula: 902.0 + (19 * 0.25) + 0.125 = 906.875 — verified below.
# The Meshtastic firmware comment saying "channel_num 20" uses a different
# counting convention (1-indexed or offset from a sub-band boundary).
DEFAULT_SLOT_INDEX: int = 19
DEFAULT_SLOT_CENTER_MHZ: float = 906.875  # empirically confirmed, see above


def slot_center_mhz(slot: int) -> float:
    """Return the center frequency for a given slot index."""
    return US915_BASE_MHZ + (slot * US915_CHANNEL_SPACING_MHZ) + US915_HALF_SPACING_MHZ


# Build the full slot table at import time so lookups are O(1).
# Maps slot_index → center_freq_mhz
SLOT_TABLE: dict[int, float] = {n: slot_center_mhz(n) for n in range(SLOT_COUNT)}

# Inverted table for freq → slot lookups (rounded to 6 decimal places)
_FREQ_TO_SLOT: dict[float, int] = {round(v, 6): k for k, v in SLOT_TABLE.items()}


def freq_to_slot(center_mhz: float) -> int:
    """
    Map a measured center frequency to the nearest slot index.

    Snaps to the 250 kHz grid rather than requiring an exact match, so small
    frequency estimation errors from the STFT centroid don't produce misses.
    """
    slot = round((center_mhz - US915_BASE_MHZ - US915_HALF_SPACING_MHZ) / US915_CHANNEL_SPACING_MHZ)
    return max(0, min(slot, SLOT_COUNT - 1))


# ---------------------------------------------------------------------------
# Named channel configurations
# All default Meshtastic presets share the same slot (DEFAULT_SLOT_INDEX) and
# are distinguished only by BW (and SF, which we cannot recover without a full
# LoRa decode in v1).  SF is listed here for documentation; we don't use it
# in detection logic.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NamedConfig:
    name: str
    slot: int
    bw_khz: int         # nominal bandwidth per spec
    sf: Optional[int]   # spreading factor — None means "unknown / not decoded"
    cr: str = "4/5"     # coding rate (all Meshtastic presets use 4/5)


NAMED_CONFIGS: list[NamedConfig] = [
    NamedConfig("LongFast",  DEFAULT_SLOT_INDEX, 250, 11),
    NamedConfig("LongSlow",  DEFAULT_SLOT_INDEX, 125, 12),
    NamedConfig("MedFast",   DEFAULT_SLOT_INDEX, 250,  9),
    NamedConfig("MedSlow",   DEFAULT_SLOT_INDEX, 250, 10),
    NamedConfig("ShortFast", DEFAULT_SLOT_INDEX, 250,  7),
    NamedConfig("ShortSlow", DEFAULT_SLOT_INDEX, 250,  8),
    NamedConfig("VLongSlow", DEFAULT_SLOT_INDEX, 125, 12),
]

# BW classification thresholds (in kHz).
# The IQ-derived bandwidth estimate is noisy, so we use a midpoint threshold
# rather than exact matching.  Everything below this is called 125 kHz;
# everything at or above is called 250 kHz.
BW_THRESHOLD_KHZ: float = 180.0


def classify_bw(detected_bw_khz: float) -> int:
    """Return 125 or 250 depending on which nominal BW class this falls into."""
    return 125 if detected_bw_khz < BW_THRESHOLD_KHZ else 250


def resolve_config_label(slot: int, bw_khz: int) -> str:
    """
    Return a human-readable config label.

    We can narrow to BW class but not SF without full demodulation, so default-
    slot detections are labeled as a BW group rather than a single named preset.
    Non-default slots include their BW class so operators can tell LongFast-class
    sub-networks (250k) from LongSlow-class ones (125k).
    """
    if slot == DEFAULT_SLOT_INDEX:
        return f"Named ({bw_khz}k)"
    return f"Non-default ({bw_khz}k)"


# ---------------------------------------------------------------------------
# SDR / capture parameters
# ---------------------------------------------------------------------------

# Center frequency for the RSPduo tuner — we tune to the middle of the band
# so the 8 MHz IF BW window covers ~903–911 MHz (slots 4–35).  Slot 0
# (902.125 MHz) falls just outside this window; move center to 906 MHz to
# include it at the cost of losing the upper end (slots 32+).
# Scanning the full 26 MHz span requires either two captures (RSPduo
# dual-tuner) or a sweep — a v2 feature.
SDR_CENTER_FREQ_MHZ: float = 906.0
# 906 MHz centre → 902–910 MHz capture window, which includes:
#   slot 0 (902.125 MHz, displayed as Slot 1) — low-usage edge of band
#   slot 8 (904.125 MHz, displayed as Slot 9) — Fairfax County LongFast sub-network
#   slot 19 (906.875 MHz, displayed as Slot 20) — default Meshtastic channel
# RSPduo single-tuner mode valid rates: 8 MHz only (6 and 2 MHz rejected by
# Init in single-tuner mode on API v3.15.1; 10 MHz also invalid).  8 Msps
# with BW_8_000 IF filter is the only combination that Init accepts.
SDR_SAMPLE_RATE_MSPS: float = 8.0      # 8 Msps → 8 MHz alias-free BW
SDR_IF_BW_MHZ: float = 8.0            # RSP IF filter width (matches sample rate)
SDR_GAIN_DB: int = 59                  # IF gain reduction (20–59 dB); raise to prevent ADC saturation
SDR_LNA_STATE: int = 3                 # LNA attenuation state (0=max gain); raise near strong local transmitters
SDR_ANTENNA_PORT: str = "A"            # RSPduo antenna port (A or B)
SDR_DEVICE_SERIAL: str = ""            # empty → use first detected RSPduo

# How long to run the capture loop.  0 means run indefinitely until SIGINT.
# Any positive integer stops cleanly after that many seconds and writes a
# session-closed marker to the DB so the dashboard can show session boundaries.
SCAN_DURATION_SECONDS: int = 0


# ---------------------------------------------------------------------------
# Burst detection thresholds
# ---------------------------------------------------------------------------

# STFT window size in samples.  Smaller → better time resolution, worse freq
# resolution.  At 8 Msps, 1024 samples = 0.128 ms per frame, freq resolution
# = 7.81 kHz/bin — resolves 125 kHz bursts across ~16 bins.
STFT_WINDOW_SIZE: int = 1024

# Overlap between consecutive STFT frames as a fraction of STFT_WINDOW_SIZE.
# 0.75 overlap (75%) gives 4× time oversampling — good for catching short
# LoRa chirp bursts without missing the leading edge.
STFT_OVERLAP_FRACTION: float = 0.75

# Energy threshold: a bin must exceed the estimated noise floor by at least
# this many dB to be considered a candidate burst bin.
# At 12 dB above the 10th-percentile floor, noise false-positives are rare
# but a real LoRa signal at this SNR sustains a continuous candidate rather
# than flickering above/below the threshold and fragmenting into sub-200-frame
# pieces that get dropped by BURST_MIN_FRAMES.  The CHIRP_MIN_STREAK_FRAMES
# gate provides false-positive rejection downstream.
# Raise toward 20 if too many false positives appear; lower toward 8 for
# weak/distant signals in a very quiet RF environment.
ENERGY_THRESHOLD_DB: float = 39.0

# Chirp confirmation: for a detection to be confirmed as a LoRa chirp (vs.
# a narrowband interferer), the burst centroid must shift by at least this
# many kHz across consecutive STFT frames.  LoRa chirps sweep the full BW
# per symbol period; a 250 kHz BW at SF11 sweeps ~0.4 kHz/frame at our
# frame rate.  Set conservatively to flag obvious chirps only.
# Slowest Meshtastic mode (LongSlow SF12, BW=125k) sweeps 0.122 kHz/frame at
# 8 Msps / 32 μs per frame.  0.08 sits below that — catches all modes while
# rejecting CW interferers whose centroid doesn't move.
CHIRP_MIN_SLOPE_KHZ_PER_FRAME: float = 0.08
# LongSlow (SF12, BW=125k) sweeps at 0.122 kHz/frame at 8 Msps / 32 μs per
# frame.  0.08 sits below that — catches all real Meshtastic modes while
# rejecting CW interferers whose centroid doesn't move at all.

# Number of consecutive same-direction centroid shifts required to confirm a
# chirp.  LoRa is strictly monotonic within each symbol period: LongFast has
# 256 frames/symbol (→ streaks of 256 between wraps), LongSlow has 1024.
# Pure random noise rarely sustains 20 consecutive same-sign shifts in a
# 200-frame window (expected longest run ≈ log2(200) ≈ 7.6 frames).
CHIRP_MIN_STREAK_FRAMES: int = 20

# BW_KHZ is no longer used directly in burst detection (BW is now inferred
# from centroid velocity); kept for reference and slot_mapper compatibility.
BURST_MIN_BW_KHZ: float = 80.0

# Minimum STFT frames a burst must accumulate before emission.
# At 8 Msps / 32 μs per frame, 200 frames = 6.4 ms.  The shortest real
# Meshtastic preamble (ShortFast SF7) is 8 symbols × 512 μs = 4.1 ms =
# ~128 frames — just below this threshold.  All standard presets use longer
# preambles (≥ 8 symbols × their symbol period), so 200 frames is safe.
# Raising from 3 ensures CHIRP_MIN_STREAK_FRAMES (20) can actually be met.
BURST_MIN_FRAMES: int = 200

# Rolling noise floor estimation window in seconds.  The noise floor estimate
# is updated this often from the lowest-energy percentile of recent FFT frames.
NOISE_FLOOR_UPDATE_INTERVAL_S: float = 1.0

# Percentile used for noise floor estimation (e.g., 10th percentile of recent
# per-bin power values).  Low percentile tracks the quietest bins, not the
# average, so intermittent signals don't inflate the floor estimate.
NOISE_FLOOR_PERCENTILE: float = 10.0


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH: str = "meshscan.db"

# SQLite WAL mode is enabled at connection time in database.py.
# Page size is set at DB creation; 4096 is optimal for the Pi 5's eMMC/SD.
DB_PAGE_SIZE: int = 4096


# ---------------------------------------------------------------------------
# Flask web dashboard
# ---------------------------------------------------------------------------

FLASK_HOST: str = "0.0.0.0"   # bind to all interfaces so Pi is reachable on LAN
FLASK_PORT: int = 5050
FLASK_DEBUG: bool = False

# How often the browser polls for new data (milliseconds).
# SSE push is preferred; this is the fallback polling interval.
UI_POLL_INTERVAL_MS: int = 2000

# Activity heatmap rolling window (minutes)
HEATMAP_WINDOW_MINUTES: int = 10

# "Active slot" definition — a slot is considered active if it had a hit
# within this many seconds of the current time.
ACTIVE_SLOT_WINDOW_SECONDS: int = 300   # 5 minutes
