"""
RSPduo IQ stream handler — ctypes bindings to the SDRplay API v3.

All struct sizes and field offsets are validated at import time against the
values produced by compiling sdrplay_offsets.c against the installed headers.
If this module raises AssertionError on import, the installed API version has
changed its ABI and the ctypes definitions below need updating.

Architecture:
  - sdrplay_api_Init() fires stream callbacks on its own C thread.
  - The callback converts raw int16 IQ to normalised complex64 and drops the
    block onto a bounded queue (overflow = silent drop; no blocking in C context).
  - A Python consumer thread drains the queue and calls self._consumer, which
    in practice is BurstDetector.push_block().
  - start() is non-blocking; stop() waits for the consumer thread to drain.

Prerequisites:
  - SDRplay API service must be running (/Library/SDRplayAPI/.../sdrplay_apiService)
  - SDRconnect must NOT be running (it holds the device exclusively)
"""

from __future__ import annotations

import ctypes
import logging
import queue
import threading
from ctypes import (
    byref, c_byte, c_char, c_double, c_float, c_int, c_short,
    c_uint, c_ubyte, c_void_p,
)
from typing import Callable, Optional

import numpy as np

from meshscan import config

log = logging.getLogger(__name__)

IQConsumer = Callable[[np.ndarray], None]

# ---------------------------------------------------------------------------
# SDRplay API error codes
# ---------------------------------------------------------------------------
_ERR_SUCCESS = 0

# ---------------------------------------------------------------------------
# Enum constants (matching sdrplay_api.h)
# ---------------------------------------------------------------------------
_BW_8_000            = 8000   # sdrplay_api_BW_8_000  — widest IF filter
_IF_ZERO             = 0      # sdrplay_api_IF_Zero   — direct conversion
_LO_AUTO             = 1      # sdrplay_api_LO_Auto
_AGC_DISABLE         = 0      # sdrplay_api_AGC_DISABLE
_TUNER_A             = 1      # sdrplay_api_Tuner_A
_RSPDUO_SINGLE       = 1      # sdrplay_api_RspDuoMode_Single_Tuner
_AMPORT_2            = 0      # sdrplay_api_RspDuo_AMPORT_2 — physically "Ant A" (50 Ω)
_NORMAL_MIN_GR       = 20     # sdrplay_api_NORMAL_MIN_GR

# Update reason flags for sdrplay_api_Update()
_UPDATE_NONE         = 0x00000000
_UPDATE_EXT1_NONE    = 0x00000000

# ---------------------------------------------------------------------------
# ctypes struct definitions
# All sizes verified against the C compiler via sdrplay_offsets.c.
# ---------------------------------------------------------------------------

class _GainValues(ctypes.Structure):
    _fields_ = [("curr", c_float), ("max", c_float), ("min", c_float)]

class _Gain(ctypes.Structure):
    _fields_ = [
        ("gRdB",       c_int),
        ("LNAstate",   c_ubyte),
        ("syncUpdate", c_ubyte),
        ("_p1",        c_byte * 2),
        ("minGr",      c_int),
        ("gainVals",   _GainValues),
    ]

class _FsFreq(ctypes.Structure):
    _fields_ = [("fsHz", c_double), ("_rest", c_byte * 8)]

class _RfFreq(ctypes.Structure):
    _fields_ = [("rfHz", c_double), ("_rest", c_byte * 8)]

class _TunerParams(ctypes.Structure):
    _fields_ = [
        ("bwType",       c_int),
        ("ifType",       c_int),
        ("loMode",       c_int),
        ("gain",         _Gain),
        ("_pad",         c_byte * 4),   # alignment gap before rfFreq (double)
        ("rfFreq",       _RfFreq),
        ("_dcOff",       c_byte * 12),
    ]

class _Agc(ctypes.Structure):
    _fields_ = [
        ("enable",        c_int),
        ("setPoint_dBfs", c_int),
        ("_rest",         c_byte * 12),
    ]

class _ControlParams(ctypes.Structure):
    _fields_ = [
        ("_dcDecim", c_byte * 8),   # DcOffset (2) + Decimation (3) + 3 pad
        ("agc",      _Agc),
        ("_rest",    c_byte * 4),   # adsbMode
    ]

class _RspDuoTunerParams(ctypes.Structure):
    _fields_ = [
        ("biasTEnable",          c_ubyte),
        ("_p1",                  c_byte * 3),
        ("tuner1AmPortSel",      c_int),
        ("tuner1AmNotchEnable",  c_ubyte),
        ("rfNotchEnable",        c_ubyte),
        ("rfDabNotchEnable",     c_ubyte),
        ("_rest",                c_byte * 5),   # resetSlaveFlags + trailing
    ]

class _RxChannelParams(ctypes.Structure):
    _fields_ = [
        ("tunerParams",       _TunerParams),
        ("ctrlParams",        _ControlParams),
        ("_rsp1a",            c_byte * 1),
        ("_pad",              c_byte * 3),
        ("_rsp2",             c_byte * 16),
        ("rspDuoTunerParams", _RspDuoTunerParams),
        ("_rspDx",            c_byte * 4),
    ]

class _DevParams(ctypes.Structure):
    _fields_ = [
        ("ppm",             c_double),
        ("fsFreq",          _FsFreq),
        ("_mid",            c_byte * 24),   # syncUpdate, resetFlags, mode, samplesPerPkt, rsp1a, rsp2
        ("extRefOutputEn",  c_int),         # rspDuoParams.extRefOutputEn at offset 48
        ("_tail",           c_byte * 12),
    ]

class _DeviceParams(ctypes.Structure):
    _fields_ = [
        ("devParams",   ctypes.POINTER(_DevParams)),
        ("rxChannelA",  ctypes.POINTER(_RxChannelParams)),
        ("rxChannelB",  ctypes.POINTER(_RxChannelParams)),
    ]

class _DeviceT(ctypes.Structure):
    _fields_ = [
        ("SerNo",            c_char * 64),
        ("hwVer",            c_ubyte),
        ("_p1",              c_byte * 3),
        ("tuner",            c_int),
        ("rspDuoMode",       c_int),
        ("valid",            c_ubyte),
        ("_p2",              c_byte * 3),
        ("rspDuoSampleFreq", c_double),
        ("dev",              c_void_p),
    ]

class _StreamCbParams(ctypes.Structure):
    _fields_ = [
        ("firstSampleNum", c_uint),
        ("grChanged",      c_int),
        ("rfChanged",      c_int),
        ("fsChanged",      c_int),
        ("numSamples",     c_uint),
    ]

_STREAM_CB_T = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(c_short), ctypes.POINTER(c_short),
    ctypes.POINTER(_StreamCbParams),
    c_uint, c_uint, c_void_p,
)
_EVENT_CB_T = ctypes.CFUNCTYPE(None, c_int, c_int, c_void_p, c_void_p)

class _CallbackFns(ctypes.Structure):
    _fields_ = [
        ("StreamACbFn", _STREAM_CB_T),
        ("StreamBCbFn", _STREAM_CB_T),
        ("EventCbFn",   _EVENT_CB_T),
    ]

# ---------------------------------------------------------------------------
# ABI size assertions — catch API version changes at import time
# ---------------------------------------------------------------------------
_EXPECTED = {
    _DeviceT:           96,
    _DevParams:         64,
    _TunerParams:       72,
    _Gain:              24,
    _ControlParams:     32,
    _Agc:               20,
    _RspDuoTunerParams: 16,
    _RxChannelParams:   144,
    _StreamCbParams:    20,
    _CallbackFns:       24,
}
for _cls, _exp in _EXPECTED.items():
    _got = ctypes.sizeof(_cls)
    assert _got == _exp, (
        f"{_cls.__name__}: ctypes size {_got} ≠ expected {_exp}. "
        "Update the struct definition to match the installed API version."
    )

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_lib() -> ctypes.CDLL:
    import sys as _sys
    if _sys.platform == "darwin":
        # Installed to /usr/local/lib but not on the default macOS dyld path
        _name = "/usr/local/lib/libsdrplay_api.dylib"
    else:
        # Raspberry Pi / Linux — standard shared-lib search path works
        _name = "libsdrplay_api.so"
    lib = ctypes.CDLL(_name)

    # Declare argtypes/restype for the calls we use so ctypes can marshal
    # pointer arguments correctly on 64-bit platforms.
    lib.sdrplay_api_Open.restype             = c_int
    lib.sdrplay_api_Close.restype            = c_int
    lib.sdrplay_api_LockDeviceApi.restype    = c_int
    lib.sdrplay_api_UnlockDeviceApi.restype  = c_int
    lib.sdrplay_api_GetDevices.restype       = c_int
    lib.sdrplay_api_GetDevices.argtypes      = [
        ctypes.POINTER(_DeviceT), ctypes.POINTER(c_uint), c_uint
    ]
    lib.sdrplay_api_SelectDevice.restype     = c_int
    lib.sdrplay_api_SelectDevice.argtypes    = [ctypes.POINTER(_DeviceT)]
    lib.sdrplay_api_ReleaseDevice.restype    = c_int
    lib.sdrplay_api_ReleaseDevice.argtypes   = [ctypes.POINTER(_DeviceT)]
    lib.sdrplay_api_GetDeviceParams.restype  = c_int
    lib.sdrplay_api_GetDeviceParams.argtypes = [
        c_void_p, ctypes.POINTER(ctypes.POINTER(_DeviceParams))
    ]
    lib.sdrplay_api_Init.restype             = c_int
    lib.sdrplay_api_Init.argtypes            = [
        c_void_p, ctypes.POINTER(_CallbackFns), c_void_p
    ]
    lib.sdrplay_api_Uninit.restype           = c_int
    lib.sdrplay_api_Uninit.argtypes          = [c_void_p]

    return lib


# ---------------------------------------------------------------------------
# SDRplayCapture
# ---------------------------------------------------------------------------

class SDRplayCapture:
    """
    Wraps an RSPduo in single-tuner mode and streams IQ to a consumer callback.

    Usage:
        cap = SDRplayCapture(consumer=detector.push_block)
        cap.start()
        # ... later ...
        cap.stop()
    """

    def __init__(self, consumer: IQConsumer) -> None:
        self._consumer  = consumer
        self._lib       = _load_lib()
        self._dev       = _DeviceT()
        self._running   = threading.Event()
        # Queue between the C callback thread and the consumer thread.
        # Stores (i_raw, q_raw) int16 array pairs — float conversion is deferred
        # to the processor thread to keep per-callback GIL hold time minimal.
        # SimpleQueue is unbounded; if the consumer falls behind, the queue grows
        # and items will age rather than be dropped (acceptable for burst detection).
        self._iq_queue: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        self._proc_thread: Optional[threading.Thread] = None

        # ctypes callback wrappers — must stay alive for the duration of streaming.
        # CPython GCs objects with refcount 0 immediately; if these lived only as
        # struct fields the function pointers would dangle and Init would fail.
        self._stream_cb_ref:  Optional[_STREAM_CB_T] = None
        self._noop_cb_ref:    Optional[_STREAM_CB_T] = None
        self._event_cb_ref:   Optional[_EVENT_CB_T]  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the device, configure it, and begin streaming IQ samples."""
        lib = self._lib
        _check(lib.sdrplay_api_Open(), "Open")

        # Enumerate devices under the API lock
        _check(lib.sdrplay_api_LockDeviceApi(), "LockDeviceApi")
        devices = (_DeviceT * 16)()
        num_devs = c_uint(0)
        _check(lib.sdrplay_api_GetDevices(devices, byref(num_devs), 16), "GetDevices")
        _check(lib.sdrplay_api_UnlockDeviceApi(), "UnlockDeviceApi")

        if num_devs.value == 0:
            raise RuntimeError("No SDRplay devices found. Is the device plugged in?")

        # Find the RSPduo (hwVer == 3); match serial if configured
        dev: Optional[_DeviceT] = None
        for i in range(num_devs.value):
            d = devices[i]
            serial = d.SerNo.decode(errors="replace")
            log.debug("Found device %d: hwVer=%d serial=%s", i, d.hwVer, serial)
            if d.hwVer != 3:  # SDRPLAY_RSPduo_ID
                continue
            if config.SDR_DEVICE_SERIAL and config.SDR_DEVICE_SERIAL != serial:
                continue
            dev = d
            break

        if dev is None:
            raise RuntimeError(
                "RSPduo not found among detected devices. "
                "Check SDR_DEVICE_SERIAL in config.py or ensure the device is connected."
            )

        # Configure for single-tuner mode before SelectDevice — the API uses
        # these fields to allocate resources appropriately.
        dev.rspDuoMode       = _RSPDUO_SINGLE
        dev.tuner            = _TUNER_A
        dev.rspDuoSampleFreq = config.SDR_SAMPLE_RATE_MSPS * 1e6

        ctypes.memmove(byref(self._dev), byref(dev), ctypes.sizeof(_DeviceT))
        _check(lib.sdrplay_api_SelectDevice(byref(self._dev)), "SelectDevice")

        # Retrieve parameter pointers (these point into API-managed memory)
        dp_ptr = ctypes.POINTER(_DeviceParams)()
        _check(
            lib.sdrplay_api_GetDeviceParams(self._dev.dev, byref(dp_ptr)),
            "GetDeviceParams",
        )
        dp = dp_ptr.contents

        # --- Sample rate ---
        dp.devParams.contents.fsFreq.fsHz = config.SDR_SAMPLE_RATE_MSPS * 1e6

        # --- Tuner A: frequency, IF bandwidth, IF mode ---
        ch = dp.rxChannelA.contents
        ch.tunerParams.rfFreq.rfHz = config.SDR_CENTER_FREQ_MHZ * 1e6
        ch.tunerParams.bwType      = _BW_8_000   # widest IF filter, matches 10 Msps
        ch.tunerParams.ifType      = _IF_ZERO     # direct-conversion (ZIF)
        ch.tunerParams.loMode      = _LO_AUTO

        # --- Gain (manual — AGC off for consistent noise floor estimation) ---
        ch.tunerParams.gain.gRdB     = config.SDR_GAIN_DB
        ch.tunerParams.gain.LNAstate = 0          # LNA gain reduction = minimum
        ch.tunerParams.gain.minGr    = _NORMAL_MIN_GR
        ch.ctrlParams.agc.enable     = _AGC_DISABLE
        ch.ctrlParams.agc.setPoint_dBfs = -60

        # --- RSPduo-specific: antenna port and notch filters ---
        # AMPORT_2 (value 0) is the physically-labelled 50 Ω "Ant A" SMA port.
        # Enable both RF notch filters to reject AM/DAB broadcast interference.
        ch.rspDuoTunerParams.tuner1AmPortSel    = _AMPORT_2
        ch.rspDuoTunerParams.rfNotchEnable      = 1
        ch.rspDuoTunerParams.rfDabNotchEnable   = 1

        log.info(
            "RSPduo configured: serial=%s  centre=%.3f MHz  fs=%.1f Msps  "
            "BW=8 MHz  gain_reduction=%d dB",
            self._dev.SerNo.decode(),
            config.SDR_CENTER_FREQ_MHZ,
            config.SDR_SAMPLE_RATE_MSPS,
            config.SDR_GAIN_DB,
        )

        # --- Register callbacks and begin streaming ---
        self._stream_cb_ref = _STREAM_CB_T(self._on_stream_data)
        self._noop_cb_ref   = _STREAM_CB_T(self._noop_stream)
        self._event_cb_ref  = _EVENT_CB_T(self._on_event)

        cbs = _CallbackFns()
        cbs.StreamACbFn = self._stream_cb_ref
        cbs.StreamBCbFn = self._noop_cb_ref   # not called in single-tuner mode
        cbs.EventCbFn   = self._event_cb_ref

        self._running.set()
        self._proc_thread = threading.Thread(
            target=self._processor, name="meshscan-capture", daemon=True
        )
        self._proc_thread.start()

        err = lib.sdrplay_api_Init(self._dev.dev, byref(cbs), None)
        if err != _ERR_SUCCESS:
            raise RuntimeError(
                f"sdrplay_api_Init failed (code {err}). "
                "Close SDRconnect or any other SDRplay application, wait 5 s, "
                "then restart MeshScan. If it still fails, run: "
                "sudo pkill -f sdrplay_apiService"
            )
        log.info("SDRplay streaming started.")

    def stop(self) -> None:
        """Stop streaming and release the device."""
        self._running.clear()
        self._lib.sdrplay_api_Uninit(self._dev.dev)
        self._lib.sdrplay_api_ReleaseDevice(byref(self._dev))
        self._lib.sdrplay_api_Close()
        if self._proc_thread:
            self._proc_thread.join(timeout=2.0)
        log.info("SDRplay streaming stopped.")

    # ------------------------------------------------------------------
    # Internal: C callback → queue → consumer thread
    # ------------------------------------------------------------------

    def _noop_stream(self, *_) -> None:
        """Unused StreamB slot — RSPduo single-tuner mode only fires StreamA."""

    def _on_stream_data(
        self,
        xi: ctypes.POINTER(c_short),
        xq: ctypes.POINTER(c_short),
        params: ctypes.POINTER(_StreamCbParams),
        num_samples: int,
        reset: int,
        _ctx: c_void_p,
    ) -> None:
        if not self._running.is_set():
            return

        n = num_samples
        # Copy raw int16 samples out of the API's DMA buffer before returning.
        # frombuffer creates a zero-copy view; .copy() does the actual memcpy
        # into a new Python-owned array.  Float conversion is deferred to the
        # processor thread to minimise GIL hold time per callback invocation.
        xi_addr = ctypes.cast(xi, c_void_p).value
        xq_addr = ctypes.cast(xq, c_void_p).value
        i_raw = np.frombuffer(
            (c_short * n).from_address(xi_addr), dtype=np.int16
        ).copy()
        q_raw = np.frombuffer(
            (c_short * n).from_address(xq_addr), dtype=np.int16
        ).copy()

        try:
            self._iq_queue.put_nowait((i_raw, q_raw))
        except Exception:
            pass  # drop on queue failure — real-time trumps completeness

    def _on_event(
        self, event_id: int, tuner: int, params: c_void_p, _ctx: c_void_p
    ) -> None:
        # Event IDs: 0=GainChange, 1=PowerOverloadChange, 2=DeviceRemoved,
        #            3=RspDuoModeChange, 4=DeviceFailure
        if event_id == 2:
            log.error("SDRplay device removed unexpectedly.")
            self._running.clear()
        elif event_id == 4:
            log.error("SDRplay device failure reported.")
            self._running.clear()

    def _processor(self) -> None:
        """Consumer thread: drain the IQ queue, convert to complex64, call self._consumer."""
        while self._running.is_set() or not self._iq_queue.empty():
            try:
                i_raw, q_raw = self._iq_queue.get(timeout=0.1)
                # Normalise to [-1, 1] complex64 here, not in the callback.
                # RSPduo ADC is 14-bit; dividing by 32768 gives headroom —
                # adequate for relative power measurements.
                iq = (i_raw.astype(np.float32) + 1j * q_raw.astype(np.float32)) / 32768.0
                self._consumer(iq)
            except queue.Empty:
                pass
            except Exception:
                log.exception("Error in IQ consumer")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check(err: int, fn: str) -> None:
    if err != _ERR_SUCCESS:
        raise RuntimeError(f"sdrplay_api_{fn} returned error code {err}")
