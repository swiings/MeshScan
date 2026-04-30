"""
STFT-based burst detector with slot-keyed chirp tracking.

Pipeline per IQ block:
  1. Compute STFT frames (windowed FFT, 75% overlap)
  2. Estimate per-bin noise floor (rolling low-percentile buffer)
  3. Map every hot bin to its nearest Meshtastic slot via _bin_to_slot
  4. Accumulate a per-slot candidate: centroid history, peak power
  5. Close a candidate when its slot goes quiet for _CANDIDATE_GAP_FRAMES
  6. On close: confirm chirp signature, classify BW from centroid velocity
  7. Emit BurstEvent for confirmed detections

Why slot-keyed instead of bin-proximity tracking:
  LoRa chirps appear as a narrow (~1 bin) peak that sweeps slowly across the
  slot's bandwidth.  At 8 Msps / 1024-sample STFT the bin width is 7.8 kHz;
  LongFast (SF11, BW=250k) sweeps at 0.977 kHz/frame and LongSlow (SF12,
  BW=125k) at 0.122 kHz/frame — both well under one bin per frame.  Keying
  on the slot index rather than instantaneous bin position:
    - Correctly accumulates all frames of a slow-sweeping chirp into one burst
    - Avoids fragmentation when the chirp's centroid drifts into an adjacent bin
    - Lets BW classification use centroid BIN-CROSSING RATE (physically meaningful)
      instead of instantaneous spectral span (always ~1 bin regardless of chirp BW)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from meshscan import config

log = logging.getLogger(__name__)

# Frames kept in the rolling noise-floor buffer.
# 512 frames × 32 μs/frame ≈ 16 ms — enough for stable low-percentile without
# storing more than ~2 MB of FFT frames.
_NOISE_BUF_FRAMES: int = 512

# A candidate closes if its slot produces no hot bins for this many frames.
# 2 frames = 64 μs — bridging a single missed frame while still closing
# promptly between packets.
_CANDIDATE_GAP_FRAMES: int = 2

# Bin-crossing rate threshold (crossings per frame) that divides 125 kHz modes
# from 250 kHz modes at 8 Msps / 1024-sample STFT (bin width = 7.81 kHz).
#
# The LoRa chirp dwells in each STFT bin for multiple frames before jumping —
# 8 frames/bin for LongFast (SF11) and 64 frames/bin for LongSlow (SF12).
# Median centroid velocity is therefore dominated by intra-bin drift (~0 kHz)
# and fails to distinguish BW.  Counting how often the centroid crosses a bin
# boundary (shift > 0.5 bins, < wrap threshold) gives a stable rate:
#   LongFast  (SF11, BW=250k): 32 crossings / 256 frames = 0.125 crossings/frame
#   LongSlow  (SF12, BW=125k): 16 crossings / 1024 frames = 0.016 crossings/frame
# Threshold of 0.05 sits cleanly between them with 2.5× margin on each side.
_BW_CROSSING_RATE_THRESHOLD: float = 0.05

# Centroid shifts larger than this are LoRa symbol-boundary wraps, not real
# velocity, and are excluded from the BW velocity calculation.
_WRAP_THRESHOLD_KHZ: float = 200.0


@dataclass
class BurstEvent:
    """A single confirmed chirp burst ready for slot mapping and DB insertion."""
    center_mhz:    float   # mean centroid over the burst (pre-snap to slot grid)
    bandwidth_khz: float   # 125 or 250, classified from centroid velocity
    peak_power_db: float   # peak excess above noise floor (proxy RSSI)
    timestamp_utc: float   # Unix timestamp of first contributing frame
    frame_count:   int     # STFT frames accumulated


BurstCallback = Callable[[BurstEvent], None]


@dataclass
class _BurstCandidate:
    """Per-slot accumulator for an in-progress burst."""
    start_time:     float
    centroids_mhz:  list[float] = field(default_factory=list)
    peak_excess_db: float       = 0.0
    last_frame_idx: int         = 0


class BurstDetector:
    """
    Consumes IQ blocks and emits BurstEvents.

    Single-threaded: all calls to push_block() must come from the same thread.
    The on_burst callback fires on that same thread — keep it fast.
    """

    def __init__(self, on_burst: BurstCallback) -> None:
        self._on_burst = on_burst

        n  = config.STFT_WINDOW_SIZE
        fs = config.SDR_SAMPLE_RATE_MSPS * 1e6

        self._n        = n
        self._hop_size = max(1, int(n * (1.0 - config.STFT_OVERLAP_FRACTION)))
        self._window   = np.hanning(n).astype(np.float32)

        # Frequency of each bin after fftshift (MHz)
        self._freq_mhz: np.ndarray = (
            np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)) / 1e6
            + config.SDR_CENTER_FREQ_MHZ
        ).astype(np.float32)

        self._bin_khz: float = (fs / n) / 1e3   # width of one STFT bin in kHz

        # Pre-compute slot index for every STFT bin so _process_frame never
        # calls freq_to_slot() in the hot path.
        self._bin_to_slot: np.ndarray = np.array(
            [config.freq_to_slot(float(f)) for f in self._freq_mhz],
            dtype=np.int32,
        )

        self._buf: np.ndarray = np.empty(0, dtype=np.complex64)

        self._noise_db: np.ndarray      = np.full(n, -200.0, dtype=np.float32)
        self._noise_buf: deque[np.ndarray] = deque(maxlen=_NOISE_BUF_FRAMES)
        self._last_floor_update: float  = 0.0
        self._warmed_up: bool           = False

        # Active candidates keyed by slot index (int)
        self._candidates: dict[int, _BurstCandidate] = {}
        self._frame_idx: int = 0

        # Diagnostic counters — logged every _STATS_INTERVAL_S seconds
        self._stats_opened:   int   = 0
        self._stats_short:    int   = 0
        self._stats_nochirp:  int   = 0
        self._stats_emitted:  int   = 0
        self._stats_last_log: float = 0.0
        self._stats_peak_excess: float = -999.0  # highest excess seen since last log

        self._cumulative_emitted: int = 0  # never reset — lifetime session total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_block(self, iq: np.ndarray) -> None:
        """Accept a block of complex64 IQ samples for processing."""
        self._buf = np.concatenate((self._buf, iq.astype(np.complex64)))
        now = time.time()
        while len(self._buf) >= self._n:
            self._process_frame(self._buf[:self._n], now)
            self._buf = self._buf[self._hop_size:]
            self._frame_idx += 1

    def get_stats(self) -> dict:
        """Return a snapshot of current detector state for the web UI."""
        noise = float(np.median(self._noise_db)) if self._warmed_up else None
        peak  = self._stats_peak_excess if self._stats_peak_excess > -999.0 else None
        thresh = config.ENERGY_THRESHOLD_DB
        cands  = len(self._candidates)

        # Recommend a threshold and explain why, based on observed detector state.
        if not self._warmed_up:
            rec_db  = 20.0
            rec_why = "Noise floor still settling — check back in a moment"
        elif cands > 12 and self._cumulative_emitted == 0:
            # Many bins hot but nothing ever confirmed → threshold is below the
            # ambient interference floor; candidates stay open indefinitely and
            # real chirp data gets diluted by noise.
            rec_db  = min(65.0, round(thresh + 4.0, 1))
            rec_why = (
                f"Many open candidates, no detections — ambient floor may exceed "
                f"{thresh:.0f} dB; try raising to {rec_db:.0f} dB"
            )
        elif self._cumulative_emitted > 0:
            rec_db  = thresh
            rec_why = "Detections confirmed — threshold is matched to this environment"
        else:
            rec_db  = 20.0
            rec_why = (
                "Default for ZIF (direct-conversion) SDRs — sits above the ~18 dB "
                "artifact floor from DC offset and IQ imbalance. Raise toward 25–30 dB "
                "near power lines or switching LED drivers."
            )

        return {
            "warmed_up":         self._warmed_up,
            "noise_floor_db":    noise,
            "threshold_db":      thresh,
            "active_candidates": cands,
            "peak_excess_db":    peak,
            "opened":            self._stats_opened,
            "short":             self._stats_short,
            "no_chirp":          self._stats_nochirp,
            "emitted":           self._stats_emitted,
            "total_emitted":     self._cumulative_emitted,
            "recommended_db":    rec_db,
            "rec_why":           rec_why,
        }

    def reset(self) -> None:
        """Clear all state (noise floor, candidates, sample buffer)."""
        self._buf = np.empty(0, dtype=np.complex64)
        self._noise_db[:] = -200.0
        self._noise_buf.clear()
        self._last_floor_update = 0.0
        self._warmed_up = False
        self._candidates.clear()
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, samples: np.ndarray, timestamp: float) -> None:
        # Windowed FFT → per-bin power in dB
        spectrum = np.fft.fftshift(np.fft.fft(samples * self._window))
        power_db = (20.0 * np.log10(np.abs(spectrum) + 1e-12)).astype(np.float32)

        # Rolling noise floor
        self._noise_buf.append(power_db)
        needs_update = (
            (not self._warmed_up and len(self._noise_buf) == _NOISE_BUF_FRAMES)
            or (self._warmed_up
                and (timestamp - self._last_floor_update) >= config.NOISE_FLOOR_UPDATE_INTERVAL_S)
        )
        if needs_update:
            self._noise_db = np.percentile(
                np.stack(self._noise_buf), config.NOISE_FLOOR_PERCENTILE, axis=0
            ).astype(np.float32)
            self._last_floor_update = timestamp
            if not self._warmed_up:
                log.info(
                    "Noise floor warmed up. median=%.1f dB  threshold=+%.0f dB",
                    float(np.median(self._noise_db)),
                    config.ENERGY_THRESHOLD_DB,
                )
            self._warmed_up = True

        if not self._warmed_up:
            return

        excess = power_db - self._noise_db
        hot    = np.where(excess > config.ENERGY_THRESHOLD_DB)[0]

        # Track peak excess for diagnostics
        if len(hot):
            self._stats_peak_excess = max(self._stats_peak_excess, float(excess[hot].max()))

        # Periodic stats log every 30 s
        if timestamp - self._stats_last_log >= 30.0:
            log.info(
                "detector stats: opened=%d  short=%d  no_chirp=%d  emitted=%d  "
                "peak_excess=%.1f dB  threshold=%.0f dB  active_candidates=%d",
                self._stats_opened, self._stats_short, self._stats_nochirp,
                self._stats_emitted, self._stats_peak_excess,
                config.ENERGY_THRESHOLD_DB, len(self._candidates),
            )
            self._stats_opened = self._stats_short = self._stats_nochirp = 0
            self._stats_emitted = 0
            self._stats_peak_excess = -999.0
            self._stats_last_log = timestamp

        if len(hot) == 0:
            self._age_candidates(active_slots=set())
            return

        # Group hot bins by slot; compute per-slot weighted centroid and peak
        slots   = self._bin_to_slot[hot]
        freqs   = self._freq_mhz[hot]
        weights = excess[hot]

        active_slots: set[int] = set()
        for slot in np.unique(slots):
            m = slots == slot
            w = weights[m]
            centroid_mhz = float(np.dot(freqs[m], w) / w.sum())
            peak_excess  = float(w.max())
            self._update_candidate(slot, centroid_mhz, peak_excess, timestamp)
            active_slots.add(int(slot))

        self._age_candidates(active_slots)

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def _update_candidate(
        self,
        slot: int,
        centroid_mhz: float,
        peak_excess: float,
        timestamp: float,
    ) -> None:
        if slot not in self._candidates:
            self._candidates[slot] = _BurstCandidate(start_time=timestamp)
            self._stats_opened += 1
        cand = self._candidates[slot]
        cand.centroids_mhz.append(centroid_mhz)
        cand.peak_excess_db  = max(cand.peak_excess_db, peak_excess)
        cand.last_frame_idx  = self._frame_idx

    def _age_candidates(self, active_slots: set[int]) -> None:
        """Close candidates whose slots had no hot bins for too long."""
        stale = [
            s for s, c in self._candidates.items()
            if s not in active_slots
            and c.last_frame_idx < self._frame_idx - _CANDIDATE_GAP_FRAMES
        ]
        for slot in stale:
            self._try_emit(slot, self._candidates.pop(slot))

    # ------------------------------------------------------------------
    # Burst confirmation and emission
    # ------------------------------------------------------------------

    def _try_emit(self, slot: int, cand: _BurstCandidate) -> None:
        frames = len(cand.centroids_mhz)
        if frames < config.BURST_MIN_FRAMES:
            self._stats_short += 1
            log.debug(
                "slot %d rejected: only %d frames (need %d)  peak=%.1f dB",
                slot + 1, frames, config.BURST_MIN_FRAMES, cand.peak_excess_db,
            )
            return

        passed, up_x, dn_x, total = self._is_chirp(cand.centroids_mhz)
        if not passed:
            self._stats_nochirp += 1
            ratio = max(up_x, dn_x) / total if total > 0 else 0.0
            log.info(
                "slot %d chirp FAIL: frames=%d peak=%.1fdB  "
                "up=%d dn=%d total=%d ratio=%.2f  (need total≥%d ratio≥0.75)",
                slot + 1, frames, cand.peak_excess_db,
                up_x, dn_x, total, ratio, config.CHIRP_MIN_STREAK_FRAMES,
            )
            return

        self._stats_emitted      += 1
        self._cumulative_emitted += 1

        # Classify BW from bin-crossing rate.
        # Count frame-to-frame centroid shifts that are large enough to represent
        # a genuine bin boundary crossing (> half a bin width) but small enough
        # not to be a symbol-boundary wrap (< _WRAP_THRESHOLD_KHZ).
        shifts_khz = [
            abs((cand.centroids_mhz[i + 1] - cand.centroids_mhz[i]) * 1000.0)
            for i in range(len(cand.centroids_mhz) - 1)
        ]
        half_bin    = self._bin_khz * 0.5
        crossings   = [s for s in shifts_khz if half_bin < s < _WRAP_THRESHOLD_KHZ]
        cross_rate  = len(crossings) / max(1, len(shifts_khz))
        bw_khz      = 250.0 if cross_rate >= _BW_CROSSING_RATE_THRESHOLD else 125.0

        self._on_burst(BurstEvent(
            center_mhz    = float(np.mean(cand.centroids_mhz)),
            bandwidth_khz = bw_khz,
            peak_power_db = cand.peak_excess_db,
            timestamp_utc = cand.start_time,
            frame_count   = len(cand.centroids_mhz),
        ))

    def _is_chirp(self, centroids_mhz: list[float]) -> tuple[bool, int, int, int]:
        """
        Returns (passed, up_crossings, dn_crossings, total_crossings).

        True if the centroid sequence shows a sustained directional frequency sweep.
        Uses directional bin-crossing dominance: at least CHIRP_MIN_STREAK_FRAMES
        crossings total, with ≥ 75% in the dominant direction.  Symbol-boundary
        wraps (|shift| ≥ _WRAP_THRESHOLD_KHZ) are excluded.
        """
        if len(centroids_mhz) < 2:
            return False, 0, 0, 0

        half_bin = self._bin_khz * 0.5

        up_x = dn_x = 0
        for i in range(len(centroids_mhz) - 1):
            s = (centroids_mhz[i + 1] - centroids_mhz[i]) * 1000.0
            if half_bin <= s < _WRAP_THRESHOLD_KHZ:
                up_x += 1
            elif -_WRAP_THRESHOLD_KHZ < s <= -half_bin:
                dn_x += 1

        total = up_x + dn_x
        if total < config.CHIRP_MIN_STREAK_FRAMES:
            return False, up_x, dn_x, total

        dominant = max(up_x, dn_x)
        return dominant / total >= 0.75, up_x, dn_x, total
