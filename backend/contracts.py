# backend/contracts.py
from __future__ import annotations

# Make imports robust even if user runs from a subfolder
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Public types live in analysis.common (shared)
from analysis.common import (
    BackendError, InvalidRequestError, DeviceError, AudioIOError, DSPError, CancelledError,
    LogEvent, ProgressEvent,
    Artifact, Result, Handle,
    # requests
    ListDevicesRequest, ReadWavRequest,
    ThdOfflineRequest, CompressorOfflineRequest, AROfflineRequest, CompareRequest,
    ThdLoopbackRequest, CompressorLoopbackRequest, ARLoopbackRequest, LoopbackRecordRequest,
)

# Offline APIs
from analysis.offline import (
    run_list_devices,
    run_wav_read,
    run_thd_offline,
    run_compressor_offline,
    run_attack_release_offline,
    run_compare,
)

# Realtime APIs
from analysis.live_measurements import (
    start_thd_loopback,
    start_compressor_loopback,
    start_attack_release_loopback,
    start_loopback_record,
)

__all__ = [
    # errors
    "BackendError", "InvalidRequestError", "DeviceError", "AudioIOError", "DSPError", "CancelledError",
    # events/models
    "LogEvent", "ProgressEvent", "Artifact", "Result", "Handle",
    # requests
    "ListDevicesRequest", "ReadWavRequest",
    "ThdOfflineRequest", "CompressorOfflineRequest", "AROfflineRequest", "CompareRequest",
    "ThdLoopbackRequest", "CompressorLoopbackRequest", "ARLoopbackRequest", "LoopbackRecordRequest",
    # offline
    "run_list_devices", "run_wav_read", "run_thd_offline", "run_compressor_offline", "run_attack_release_offline", "run_compare",
    # realtime
    "start_thd_loopback", "start_compressor_loopback", "start_attack_release_loopback", "start_loopback_record",
]
