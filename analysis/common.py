# analysis/common.py
"""
Shared types + helpers for Audio Measurement Toolkit.

This module is internal; GUI MUST import only from backend.contracts.
"""
from __future__ import annotations

import csv
import datetime as _dt
import math
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

SCHEMA = "audio_toolkit_csv_v1"

# ============================================================
# Error model
# ============================================================

class BackendError(Exception):
    """Base class for backend errors."""

class InvalidRequestError(BackendError):
    """Request parameters are invalid or required resources are missing."""

class DeviceError(BackendError):
    """Audio device configuration is invalid or device is unavailable."""

class AudioIOError(BackendError):
    """Audio play/record or WAV read/write error."""

class DSPError(BackendError):
    """DSP analysis error."""

class CancelledError(BackendError):
    """Operation cancelled via stop_event or handle.cancel()."""

# ============================================================
# Events
# ============================================================

LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
Phase = Literal["validate", "stimulus", "playrec", "streaming", "analyze", "export", "done"]

@dataclass(frozen=True)
class LogEvent:
    level: LogLevel = "INFO"
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ProgressEvent:
    phase: Phase
    percent: float | None = None
    message: str = ""
    # IMPORTANT: when phase="streaming", meta MUST include {"chunk": int, "data": dict}
    meta: dict[str, Any] = field(default_factory=dict)

# ============================================================
# Output models
# ============================================================

Feature = Literal["audio_io", "thd", "compressor", "attack_release", "compare", "loopback_record"]
Mode = Literal["offline", "loopback_realtime"]
ArtifactKind = Literal["csv", "wav"]

@dataclass(frozen=True)
class Artifact:
    kind: ArtifactKind
    path: str
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass
class Result:
    feature: Feature
    mode: Mode
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

class Handle:
    def __init__(self, thread: threading.Thread, stop_event: threading.Event, box: dict[str, Any]):
        self._t = thread
        self._stop = stop_event
        self._box = box
    def join(self, timeout: float | None = None) -> Result:
        self._t.join(timeout=timeout)
        if self._t.is_alive():
            raise BackendError("Task still running")
        exc = self._box.get("exc")
        if exc is not None:
            raise exc
        res = self._box.get("result")
        if res is None:
            raise BackendError("Task finished without result")
        return res
    def cancel(self) -> None:
        self._stop.set()
    def is_running(self) -> bool:
        return self._t.is_alive()

def _start_async(name: str, fn: Callable[[], Result], stop_event: threading.Event) -> Handle:
    box: dict[str, Any] = {}
    def worker():
        try:
            box["result"] = fn()
        except Exception as exc:
            box["exc"] = exc
    t = threading.Thread(target=worker, daemon=True, name=name)
    t.start()
    return Handle(t, stop_event, box)

# ============================================================
# Requests (match GUI imports)
# ============================================================

@dataclass(frozen=True)
class ListDevicesRequest:
    ...

@dataclass(frozen=True)
class ReadWavRequest:
    path: str

@dataclass(frozen=True)
class ThdOfflineRequest:
    wav_path: str
    freq_hz: float = 1000.0
    hmax: int = 5
    export_csv: bool = True
    out_dir: str | None = None

@dataclass(frozen=True)
class CompressorOfflineRequest:
    wav_path: str
    freq_hz: float = 1000.0
    export_csv: bool = True
    out_dir: str | None = None

@dataclass(frozen=True)
class AROfflineRequest:
    wav_path: str
    rms_win_ms: float = 5.0
    export_csv: bool = True
    out_dir: str | None = None

@dataclass(frozen=True)
class CompareRequest:
    ref_wav_path: str
    tgt_wav_path: str
    max_lag_s: float = 1.0
    export_csv: bool = True
    out_dir: str | None = None

@dataclass(frozen=True)
class ThdLoopbackRequest:
    freq_hz: float = 1000.0
    amp: float = 0.7
    hmax: int = 5
    duration_s: float = 2.0
    sample_rate: int | None = None
    channels: int = 1
    input_device: int | None = None
    output_device: int | None = None
    export_csv: bool = True
    export_wav: bool = True
    out_dir: str | None = None
    blocksize: int = 2048

@dataclass(frozen=True)
class CompressorLoopbackRequest:
    freq_hz: float = 1000.0
    amp_max: float = 1.0
    levels: list[float] | None = None
    step_duration_s: float = 0.25
    sample_rate: int | None = None
    channels: int = 1
    input_device: int | None = None
    output_device: int | None = None
    export_csv: bool = True
    export_wav: bool = True
    out_dir: str | None = None
    blocksize: int = 2048

@dataclass(frozen=True)
class ARLoopbackRequest:
    freq_hz: float = 1000.0
    amp: float = 0.7
    rms_win_ms: float = 5.0
    duration_s: float = 2.0
    sample_rate: int | None = None
    channels: int = 1
    input_device: int | None = None
    output_device: int | None = None
    export_csv: bool = True
    export_wav: bool = True
    out_dir: str | None = None
    blocksize: int = 2048

@dataclass(frozen=True)
class LoopbackRecordRequest:
    input_wav_path: str
    output_path: str | None = None
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int | None = None
    input_channels: int = 1
    output_channels: int = 1
    blocksize: int = 2048

# ============================================================
# Common helpers
# ============================================================

def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _new_run_id() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

def _default_runs_dir() -> Path:
    return Path.cwd() / "runs"

def _resolve_out_dir(out_dir: str | None, run_id: str) -> Path:
    base = Path(out_dir) if out_dir else _default_runs_dir()
    return base / run_id

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _emit_log(on_log: Callable[[LogEvent], None] | None, level: LogLevel, msg: str, **meta: Any) -> None:
    if on_log:
        on_log(LogEvent(level=level, message=msg, meta=dict(meta)))

def _emit_progress(on_progress: Callable[[ProgressEvent], None] | None, phase: Phase, *, percent: float | None = None, message: str = "", meta: dict[str, Any] | None = None) -> None:
    if on_progress:
        on_progress(ProgressEvent(phase=phase, percent=percent, message=message, meta=meta or {}))

def _as_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1)
    raise InvalidRequestError("Audio must be 1D or 2D")

def _import_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
        return sd
    except Exception as exc:
        raise DeviceError("sounddevice is required for realtime loopback. Install: pip install sounddevice") from exc

# ============================================================
# WAV I/O
# ============================================================

def _wav_read(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioIOError(f"Unsupported sampwidth={sw} bytes")
    if ch > 1:
        a = a.reshape(-1, ch)
    return int(sr), a

def _wav_write(path: Path, samples: np.ndarray, sr: int) -> None:
    x = np.asarray(samples)
    if x.ndim == 1:
        ch = 1
        flat = x
    elif x.ndim == 2:
        ch = int(x.shape[1])
        flat = x.reshape(-1)
    else:
        raise AudioIOError("samples must be 1D/2D")
    _ensure_dir(path.parent)
    y = np.clip(flat, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm)

# ============================================================
# CSV (schema v1)
# ============================================================

def _write_csv_with_meta(path: Path, meta: dict[str, Any], header: list[str], rows: list[tuple[Any, ...]]) -> None:
    _ensure_dir(path.parent)
    meta2 = dict(meta)
    meta2.setdefault("schema", SCHEMA)
    with path.open("w", newline="", encoding="utf-8") as f:
        for k, v in meta2.items():
            f.write(f"# {k}={v}\n")
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))

# ============================================================
# DSP helpers
# ============================================================

def _spectrum_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    n = int(2 ** math.ceil(math.log2(max(2048, min(len(x), 65536)))))
    w = np.hanning(min(len(x), n))
    xx = x[: len(w)] * w
    X = np.fft.rfft(xx, n)
    mag = np.abs(X) / (np.sum(w) / 2 + 1e-12)
    mag_db = 20 * np.log10(mag + 1e-15)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return f, mag_db

def _compute_thd(x: np.ndarray, sr: int, f0: float, hmax: int) -> tuple[list[tuple[float, float]], float]:
    f, db = _spectrum_db(x, sr)
    lin = 10 ** (db / 20.0)
    def peak(ff: float) -> float:
        bw = max(5.0, ff * 0.01)
        m = (f >= ff - bw) & (f <= ff + bw)
        return float(np.max(lin[m])) if np.any(m) else 0.0
    fund = peak(f0)
    harms: list[tuple[float, float]] = []
    harm_pow = 0.0
    for k in range(1, hmax + 1):
        fk = f0 * k
        ak = peak(fk)
        harms.append((fk, 20.0 * math.log10(ak + 1e-15)))
        if k >= 2:
            harm_pow += ak * ak
    thd = math.sqrt(harm_pow) / (fund + 1e-15)
    return harms, thd

def _rms_envelope_db(x: np.ndarray, sr: int, win_ms: float) -> tuple[list[float], list[float]]:
    win = max(1, int(win_ms * 1e-3 * sr))
    hop = max(1, win // 2)
    env: list[float] = []
    tt: list[float] = []
    for i in range(0, len(x) - win + 1, hop):
        seg = x[i : i + win]
        r = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        env.append(20.0 * math.log10(r + 1e-12))
        tt.append(i / sr)
    return env, tt

def _estimate_attack_release(env_db: Sequence[float], t: Sequence[float]) -> tuple[float, float]:
    if len(env_db) < 5:
        return 0.0, 0.0
    e = np.array(env_db, dtype=float)
    tt = np.array(t, dtype=float)
    base = float(np.percentile(e, 10))
    peak = float(np.percentile(e, 90))
    if peak - base < 1.0:
        return 0.0, 0.0
    lo = base + 0.1 * (peak - base)
    hi = base + 0.9 * (peak - base)
    i0 = int(np.argmax(e > lo))
    i1 = i0 + int(np.argmax(e[i0:] > hi))
    atk = (tt[i1] - tt[i0]) * 1000.0 if i1 > i0 else 0.0
    ih = int(np.where(e > hi)[0][-1]) if np.any(e > hi) else 0
    il = ih + int(np.argmax(e[ih:] < lo)) if ih < len(e) - 1 else ih
    rel = (tt[il] - tt[ih]) * 1000.0 if il > ih else 0.0
    return float(max(0.0, atk)), float(max(0.0, rel))
