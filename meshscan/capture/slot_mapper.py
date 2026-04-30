"""
Frequency → slot → named config resolver.

Takes the center_mhz and bandwidth_khz from a BurstEvent and returns:
- slot index (int)
- nominal BW class (125 or 250 kHz)
- human-readable config label string

All frequency math and label logic lives here; no numeric literals — everything
references config.py constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from meshscan import config


@dataclass
class SlotMatch:
    slot: int
    center_mhz: float      # snapped to SLOT_TABLE grid, not the raw estimate
    bw_khz: int            # classified nominal BW (125 or 250)
    label: str             # e.g. "Named (250k)", "Non-standard slot 9"
    is_default_slot: bool


def map_burst(center_mhz: float, detected_bw_khz: float) -> SlotMatch:
    """
    Map a detected burst to its Meshtastic slot and config label.

    Args:
        center_mhz: estimated center frequency from burst centroid
        detected_bw_khz: estimated spectral width from burst frequency span

    Returns:
        SlotMatch with slot index, snapped center freq, BW class, and label.
    """
    slot         = config.freq_to_slot(center_mhz)
    snapped_mhz  = config.SLOT_TABLE[slot]
    bw_khz       = config.classify_bw(detected_bw_khz)
    label        = config.resolve_config_label(slot, bw_khz)
    is_default   = slot == config.DEFAULT_SLOT_INDEX

    return SlotMatch(
        slot           = slot,
        center_mhz     = snapped_mhz,
        bw_khz         = bw_khz,
        label          = label,
        is_default_slot= is_default,
    )
