# backend/contracts.py
"""
Backend API Contract for Audio Measurement Toolkit.

NON-NEGOTIABLE
- GUI_D_3_2_1.py is the immutable reference GUI layout.
- Backend must NOT import tkinter / messagebox / depend on GUI state.
- GUI must call ONLY the public facade functions defined in this module:
    from backend.contracts import ...

API rules
- Sync (fast):   run_xxx(request) -> MeasurementResult-like object (here: Result)
- Async (long):  start_xxx(request, *, stop_event, on_progress, on_log) -> Handle(join/cancel/is_running)

Realtime streaming requirement
- During loopback streaming, backend MUST call on_progress with:
    ProgressEvent(phase="streaming", meta={"chunk": <int>, "data": <plain_dict>})
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
# Output models (minimal, GUI-friendly)
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


def _emit_progress(
    on_progress: Callable[[ProgressEvent], None] | None,
    phase: Phase,
    *,
    percent: float | None = None,
    message: str = "",
    meta: dict[str, Any] | None = None,
) -> None:
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
# WAV I/O (self-contained)
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


def _write_csv_with_meta(path: Path, meta: dict[str, Any], header: list[str], rows: list[tuple[Any, ...]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


# ============================================================
# Public API - devices
# ============================================================

def run_list_devices(request: ListDevicesRequest) -> Result:
    sd = _import_sounddevice()
    devs = sd.query_devices()
    ins: list[str] = []
    outs: list[str] = []
    for i, d in enumerate(devs):
        name = d.get("name", f"dev{i}")
        if int(d.get("max_input_channels", 0)) > 0:
            ins.append(f"{i}: {name}")
        if int(d.get("max_output_channels", 0)) > 0:
            outs.append(f"{i}: {name}")
    return Result(feature="audio_io", mode="offline", summary={"inputs": ins, "outputs": outs})


# ============================================================
# Public API - wav read
# ============================================================

def run_wav_read(request: ReadWavRequest) -> Result:
    p = Path(request.path)
    if not p.is_file():
        raise AudioIOError(f"WAV not found: {p}")
    sr, x = _wav_read(p)
    ch = 1 if x.ndim == 1 else int(x.shape[1])
    return Result(feature="audio_io", mode="offline", summary={"path": str(p), "fs": sr, "n_channels": ch, "n_samples": int(x.shape[0])})


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


# ============================================================
# Public API - offline analyses
# ============================================================

def run_thd_offline(request: ThdOfflineRequest) -> Result:
    p = Path(request.wav_path)
    if not p.is_file():
        raise InvalidRequestError("wav_path not found")

    sr, x = _wav_read(p)
    mono = _as_mono(x)
    harms, thd = _compute_thd(mono, sr, float(request.freq_hz), int(request.hmax))

    run_id = _new_run_id()
    ts = _utc_now_iso()
    outdir = _resolve_out_dir(request.out_dir, run_id)
    _ensure_dir(outdir)

    arts: list[Artifact] = []
    if request.export_csv:
        cp = outdir / "thd_harmonics.csv"
        _write_csv_with_meta(
            cp,
            {"feature": "thd", "mode": "offline", "run_id": run_id, "timestamp": ts, "sample_rate": sr, "freq_hz": request.freq_hz, "hmax": request.hmax},
            ["harmonic", "freq_hz", "mag_db"],
            [(i + 1, fk, dbv) for i, (fk, dbv) in enumerate(harms)],
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="THD harmonics"))

    return Result(
        feature="thd",
        mode="offline",
        summary={
            "fs": sr,
            "freq_hz": float(request.freq_hz),
            "hmax": int(request.hmax),
            "thd_ratio": float(thd),
            "thd_percent": float(100.0 * thd),
            "thd_db": float(20.0 * math.log10(thd + 1e-15)),
        },
        artifacts=arts,
    )


def run_compressor_offline(request: CompressorOfflineRequest) -> Result:
    p = Path(request.wav_path)
    if not p.is_file():
        raise InvalidRequestError("wav_path not found")

    sr, x = _wav_read(p)
    mono = _as_mono(x)

    frame = max(256, int(0.05 * sr))
    hop = frame
    vals: list[float] = []
    for i in range(0, len(mono) - frame + 1, hop):
        seg = mono[i : i + frame]
        r = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        vals.append(20.0 * math.log10(r + 1e-12))

    if len(vals) < 10:
        raise DSPError("signal too short for compressor analysis")

    vals_arr = np.array(vals, dtype=float)
    xlv = np.quantile(vals_arr, np.linspace(0.05, 0.95, 20))
    ylv = xlv.copy()  # placeholder curve
    slopes = np.diff(ylv) / (np.diff(xlv) + 1e-9)
    knee = int(np.argmin(slopes)) if slopes.size else 0
    thr_db = float(xlv[knee]) if xlv.size else 0.0
    ratio = float(max(1.0, 1.0 / (slopes[knee] + 1e-6))) if slopes.size else 1.0
    gain_offset_db = 0.0

    run_id = _new_run_id()
    ts = _utc_now_iso()
    outdir = _resolve_out_dir(request.out_dir, run_id)
    _ensure_dir(outdir)

    arts: list[Artifact] = []
    if request.export_csv:
        cp = outdir / "compressor_curve.csv"
        _write_csv_with_meta(
            cp,
            {"feature": "compressor", "mode": "offline", "run_id": run_id, "timestamp": ts, "sample_rate": sr, "thr_db": thr_db, "ratio": ratio, "gain_offset_db": gain_offset_db},
            ["in_level_db", "out_level_db"],
            list(zip(xlv.astype(float), ylv.astype(float))),
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="Compressor curve (heuristic)"))

    return Result(
        feature="compressor",
        mode="offline",
        summary={"fs": sr, "thr_db": thr_db, "ratio": ratio, "gain_offset_db": gain_offset_db, "no_compression": bool(ratio <= 1.05)},
        artifacts=arts,
    )


def run_attack_release_offline(request: AROfflineRequest) -> Result:
    p = Path(request.wav_path)
    if not p.is_file():
        raise InvalidRequestError("wav_path not found")

    sr, x = _wav_read(p)
    mono = _as_mono(x)

    env, tt = _rms_envelope_db(mono, sr, float(request.rms_win_ms))
    atk, rel = _estimate_attack_release(env, tt)

    run_id = _new_run_id()
    ts = _utc_now_iso()
    outdir = _resolve_out_dir(request.out_dir, run_id)
    _ensure_dir(outdir)

    arts: list[Artifact] = []
    if request.export_csv:
        cp = outdir / "ar_envelope.csv"
        _write_csv_with_meta(
            cp,
            {"feature": "attack_release", "mode": "offline", "run_id": run_id, "timestamp": ts, "sample_rate": sr, "attack_ms": atk, "release_ms": rel, "rms_win_ms": request.rms_win_ms},
            ["time_s", "envelope_db"],
            list(zip(tt, env)),
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="A/R envelope"))

    return Result(
        feature="attack_release",
        mode="offline",
        summary={"fs": sr, "rms_win_ms": float(request.rms_win_ms), "attack_ms": atk, "release_ms": rel},
        artifacts=arts,
    )


# ============================================================
# Compare (offline)
# ============================================================

def _estimate_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> int:
    x = x - np.mean(x)
    y = y - np.mean(y)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    nfft = 1 << int(math.ceil(math.log2(max(8, 2 * n))))
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    c = np.fft.irfft(X * np.conj(Y), nfft)
    c = np.concatenate([c[-(n - 1) :], c[:n]])
    mid = len(c) // 2
    lo = max(0, mid - max_lag)
    hi = min(len(c), mid + max_lag + 1)
    seg = c[lo:hi]
    return int(np.argmax(seg) - (mid - lo))


def _apply_lag(ref: np.ndarray, out: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        ref2 = ref[:-lag]
        out2 = out[lag:]
    elif lag < 0:
        l = -lag
        ref2 = ref[l:]
        out2 = out[:-l]
    else:
        ref2, out2 = ref, out
    n = min(len(ref2), len(out2))
    return ref2[:n], out2[:n]


def _estimate_gain(ref: np.ndarray, out: np.ndarray) -> float:
    denom = float(np.dot(out, out) + 1e-12)
    return float(np.dot(ref, out) / denom)


def run_compare(request: CompareRequest) -> Result:
    pr = Path(request.ref_wav_path)
    po = Path(request.tgt_wav_path)
    if not pr.is_file() or not po.is_file():
        raise InvalidRequestError("ref_wav_path/tgt_wav_path not found")

    sr_r, r = _wav_read(pr)
    sr_o, o = _wav_read(po)
    if sr_r != sr_o:
        raise InvalidRequestError("Sample rate mismatch")

    r = _as_mono(r)
    o = _as_mono(o)

    max_lag = int(max(1, float(request.max_lag_s) * sr_r))
    lag = _estimate_lag(r, o, max_lag)
    rr, oo = _apply_lag(r, o, lag)

    g = _estimate_gain(rr, oo)
    oo_g = oo / (g if g != 0 else 1.0)
    res = oo_g - rr

    rms_r = float(np.sqrt(np.mean(rr * rr) + 1e-12))
    rms_res = float(np.sqrt(np.mean(res * res) + 1e-12))
    gain_err_db = 20.0 * math.log10(abs(g) + 1e-12)
    snr_db = 20.0 * math.log10((rms_r + 1e-12) / (rms_res + 1e-12))

    run_id = _new_run_id()
    ts = _utc_now_iso()
    outdir = _resolve_out_dir(request.out_dir, run_id)
    _ensure_dir(outdir)

    arts: list[Artifact] = []
    if request.export_csv:
        cp = outdir / "compare_metrics.csv"
        _write_csv_with_meta(
            cp,
            {"feature": "compare", "mode": "offline", "run_id": run_id, "timestamp": ts, "sample_rate": sr_r, "lag_samples": lag, "gain_linear": g},
            ["metric", "value"],
            [("lag_samples", lag), ("lag_ms", 1000.0 * lag / sr_r), ("gain_error_db", gain_err_db), ("snr_db", snr_db)],
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="Compare metrics"))

    return Result(
        feature="compare",
        mode="offline",
        summary={"fs": sr_r, "lag_samples": lag, "lag_ms": 1000.0 * lag / sr_r, "gain_error_db": gain_err_db, "snr_db": snr_db},
        artifacts=arts,
    )


# ============================================================
# Realtime duplex runner (streams chunks)
# ============================================================

class _DuplexRunner:
    def __init__(
        self,
        sd,
        playback: np.ndarray,  # shape (n, out_ch)
        *,
        sr: int,
        in_ch: int,
        out_ch: int,
        input_device: int | None,
        output_device: int | None,
        blocksize: int,
        stop_event: threading.Event,
        on_progress: Callable[[ProgressEvent], None] | None,
        on_log: Callable[[LogEvent], None] | None,
        make_payload: Callable[[np.ndarray, int], dict[str, Any]] | None = None,
    ):
        self.sd = sd
        self.playback = np.asarray(playback, dtype=np.float32)
        self.sr = int(sr)
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.input_device = input_device
        self.output_device = output_device
        self.blocksize = int(blocksize)
        self.stop_event = stop_event
        self.on_progress = on_progress
        self.on_log = on_log
        self.make_payload = make_payload

        self._idx = 0
        self._chunk = 0
        self._rx: list[np.ndarray] = []
        self._exc: Exception | None = None

    def run(self) -> np.ndarray:
        def callback(indata, outdata, frames, time_info, status):
            try:
                if status:
                    _emit_log(self.on_log, "WARN", f"stream status: {status}")
                if self.stop_event.is_set():
                    raise CancelledError("Cancelled")

                end = min(self._idx + frames, self.playback.shape[0])
                out_chunk = self.playback[self._idx:end]
                if out_chunk.shape[0] < frames:
                    pad = np.zeros((frames - out_chunk.shape[0], self.out_ch), dtype=np.float32)
                    out_chunk = np.vstack([out_chunk, pad])
                outdata[:] = out_chunk

                self._rx.append(indata.copy())
                self._idx = end

                buf = np.vstack(self._rx)[:, 0] if self._rx else np.zeros((0,), dtype=np.float32)
                payload = self.make_payload(buf, self._chunk) if self.make_payload else {}
                _emit_progress(self.on_progress, "streaming", meta={"chunk": int(self._chunk), "data": payload})
                self._chunk += 1

                if self._idx >= self.playback.shape[0]:
                    raise self.sd.CallbackStop()
            except Exception as e:
                self._exc = e
                raise

        with self.sd.Stream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            device=(self.input_device, self.output_device),
            channels=(self.in_ch, self.out_ch),
            dtype="float32",
            callback=callback,
        ):
            while (not self.stop_event.is_set()) and (self._idx < self.playback.shape[0]):
                time.sleep(0.02)

        if self._exc is not None:
            if isinstance(self._exc, CancelledError):
                raise self._exc
            raise AudioIOError(str(self._exc)) from self._exc

        rx = np.vstack(self._rx) if self._rx else np.zeros((0, self.in_ch), dtype=np.float32)
        return rx


# ============================================================
# Loopback record (async)
# ============================================================

def start_loopback_record(
    request: LoopbackRecordRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> Handle:
    stop_event = stop_event or threading.Event()

    def impl() -> Result:
        sd = _import_sounddevice()
        inp = Path(request.input_wav_path)
        if not inp.is_file():
            raise InvalidRequestError("input_wav_path not found")

        sr_in, x = _wav_read(inp)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]

        sr = int(request.sample_rate or sr_in)
        if sr != sr_in:
            raise InvalidRequestError("Resampling not implemented; keep sample_rate=None or match WAV rate")

        out_ch = int(request.output_channels)
        in_ch = int(request.input_channels)

        if x.shape[1] != out_ch:
            x = np.tile(x[:, :1], (1, out_ch))

        run_id = _new_run_id()
        outdir = _resolve_out_dir(None, run_id)
        _ensure_dir(outdir)

        out_path = Path(request.output_path) if request.output_path else (outdir / "received.wav")

        _emit_log(on_log, "INFO", "Loopback record started", input_device=request.input_device, output_device=request.output_device)
        _emit_progress(on_progress, "stimulus", message="playback prepared", meta={"chunk": 0, "data": {}})

        runner = _DuplexRunner(
            sd,
            x,
            sr=sr,
            in_ch=in_ch,
            out_ch=out_ch,
            input_device=request.input_device,
            output_device=request.output_device,
            blocksize=request.blocksize,
            stop_event=stop_event,
            on_progress=on_progress,
            on_log=on_log,
            make_payload=None,
        )
        rx = runner.run()

        _emit_progress(on_progress, "export", message="writing wav", meta={"chunk": 0, "data": {}})
        _wav_write(out_path, rx, sr)

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(
            feature="loopback_record",
            mode="loopback_realtime",
            summary={"output_path": str(out_path), "fs": sr, "channels": in_ch},
            artifacts=[Artifact(kind="wav", path=str(out_path), description="Recorded loopback WAV")],
        )

    return _start_async("loopback_record", impl, stop_event)


# ============================================================
# THD loopback (async)
# ============================================================

def start_thd_loopback(
    request: ThdLoopbackRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> Handle:
    stop_event = stop_event or threading.Event()

    def impl() -> Result:
        sd = _import_sounddevice()
        if request.freq_hz <= 0:
            raise InvalidRequestError("freq_hz must be >0")
        if request.hmax < 1:
            raise InvalidRequestError("hmax must be >= 1")

        sr = int(request.sample_rate or 48000)
        ch = int(request.channels or 1)
        n = int(sr * float(request.duration_s))
        tt = np.arange(n) / sr

        tone = (float(request.amp) * np.sin(2 * np.pi * float(request.freq_hz) * tt)).astype(np.float32)
        playback = tone[:, None] if ch == 1 else np.tile(tone[:, None], (1, ch))

        run_id = _new_run_id()
        ts = _utc_now_iso()
        outdir = _resolve_out_dir(request.out_dir, run_id)
        _ensure_dir(outdir)

        def payload(buf: np.ndarray, chunk: int) -> dict[str, Any]:
            if buf.size < 2048:
                return {"freq_axis_hz": [], "mag_db": []}
            f, db = _spectrum_db(buf, sr)
            step = max(1, len(f) // 512)
            return {"freq_axis_hz": f[::step].astype(float).tolist(), "mag_db": db[::step].astype(float).tolist()}

        _emit_log(on_log, "INFO", "THD loopback started", input_device=request.input_device, output_device=request.output_device)
        _emit_progress(on_progress, "stimulus", message="tone generated", meta={"chunk": 0, "data": {}})

        runner = _DuplexRunner(
            sd,
            playback,
            sr=sr,
            in_ch=ch,
            out_ch=ch,
            input_device=request.input_device,
            output_device=request.output_device,
            blocksize=request.blocksize,
            stop_event=stop_event,
            on_progress=on_progress,
            on_log=on_log,
            make_payload=payload,
        )
        rx = runner.run()
        mono = _as_mono(rx)

        _emit_progress(on_progress, "analyze", message="compute thd", meta={"chunk": 0, "data": {}})
        harms, thd = _compute_thd(mono, sr, float(request.freq_hz), int(request.hmax))

        arts: list[Artifact] = []
        if request.export_wav:
            wp = outdir / "thd_recorded.wav"
            _emit_progress(on_progress, "export", message="export wav", meta={"chunk": 0, "data": {}})
            _wav_write(wp, rx, sr)
            arts.append(Artifact(kind="wav", path=str(wp), description="Recorded THD capture"))

        if request.export_csv:
            cp = outdir / "thd_harmonics.csv"
            _emit_progress(on_progress, "export", message="export csv", meta={"chunk": 0, "data": {}})
            _write_csv_with_meta(
                cp,
                {"feature": "thd", "mode": "loopback_realtime", "run_id": run_id, "timestamp": ts, "sample_rate": sr, "freq_hz": request.freq_hz, "hmax": request.hmax},
                ["harmonic", "freq_hz", "mag_db"],
                [(i + 1, fk, dbv) for i, (fk, dbv) in enumerate(harms)],
            )
            arts.append(Artifact(kind="csv", path=str(cp), description="THD harmonics"))

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(
            feature="thd",
            mode="loopback_realtime",
            summary={
                "fs": sr,
                "freq_hz": float(request.freq_hz),
                "hmax": int(request.hmax),
                "thd_ratio": float(thd),
                "thd_percent": float(100.0 * thd),
                "thd_db": float(20.0 * math.log10(thd + 1e-15)),
            },
            artifacts=arts,
        )

    return _start_async("thd_loopback", impl, stop_event)


# ============================================================
# Compressor loopback (async)
# ============================================================

def start_compressor_loopback(
    request: CompressorLoopbackRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> Handle:
    stop_event = stop_event or threading.Event()

    def impl() -> Result:
        sd = _import_sounddevice()
        sr = int(request.sample_rate or 48000)
        ch = int(request.channels or 1)

        levels = request.levels or list(np.linspace(0.05, float(request.amp_max), 24))
        step_n = int(sr * float(request.step_duration_s))

        sig = np.zeros(step_n * len(levels), dtype=np.float32)
        idx = 0
        for a in levels:
            tt = np.arange(step_n) / sr
            sig[idx : idx + step_n] = float(a) * np.sin(2 * np.pi * float(request.freq_hz) * tt)
            idx += step_n

        playback = sig[:, None] if ch == 1 else np.tile(sig[:, None], (1, ch))

        run_id = _new_run_id()
        ts = _utc_now_iso()
        outdir = _resolve_out_dir(request.out_dir, run_id)
        _ensure_dir(outdir)

        def payload(buf: np.ndarray, chunk: int) -> dict[str, Any]:
            win = max(64, int(0.05 * sr))
            seg = buf[-win:] if buf.size >= win else buf
            r = float(np.sqrt(np.mean(seg * seg) + 1e-12))
            return {"rms_db": 20.0 * math.log10(r + 1e-12)}

        _emit_log(on_log, "INFO", "Compressor loopback started", input_device=request.input_device, output_device=request.output_device)
        _emit_progress(on_progress, "stimulus", message="stimulus built", meta={"chunk": 0, "data": {}})

        runner = _DuplexRunner(
            sd,
            playback,
            sr=sr,
            in_ch=ch,
            out_ch=ch,
            input_device=request.input_device,
            output_device=request.output_device,
            blocksize=request.blocksize,
            stop_event=stop_event,
            on_progress=on_progress,
            on_log=on_log,
            make_payload=payload,
        )
        rx = runner.run()
        mono = _as_mono(rx)

        _emit_progress(on_progress, "analyze", message="analyze steps", meta={"chunk": 0, "data": {}})

        in_db: list[float] = []
        out_db: list[float] = []
        gr_db: list[float] = []
        for k, a in enumerate(levels):
            seg = mono[k * step_n : (k + 1) * step_n]
            rin = 20.0 * math.log10(abs(float(a)) / math.sqrt(2) + 1e-12)
            rout = 20.0 * math.log10(float(np.sqrt(np.mean(seg * seg) + 1e-12)) + 1e-12)
            in_db.append(rin)
            out_db.append(rout)
            gr_db.append(max(0.0, rin - rout))

        xlv = np.array(in_db, dtype=float)
        ylv = np.array(out_db, dtype=float)
        slopes = np.diff(ylv) / (np.diff(xlv) + 1e-9)
        knee = int(np.argmin(slopes)) if slopes.size else 0
        thr_db = float(xlv[knee]) if xlv.size else 0.0
        ratio = float(max(1.0, 1.0 / (slopes[knee] + 1e-6))) if slopes.size else 1.0
        gain_offset_db = float(np.median(ylv - xlv)) if xlv.size else 0.0

        arts: list[Artifact] = []
        if request.export_wav:
            wp = outdir / "compressor_recorded.wav"
            _emit_progress(on_progress, "export", message="export wav", meta={"chunk": 0, "data": {}})
            _wav_write(wp, rx, sr)
            arts.append(Artifact(kind="wav", path=str(wp), description="Recorded compressor capture"))

        if request.export_csv:
            cp = outdir / "compressor_curve.csv"
            _emit_progress(on_progress, "export", message="export csv", meta={"chunk": 0, "data": {}})
            _write_csv_with_meta(
                cp,
                {"feature": "compressor", "mode": "loopback_realtime", "run_id": run_id, "timestamp": ts, "sample_rate": sr, "thr_db": thr_db, "ratio": ratio, "gain_offset_db": gain_offset_db},
                ["in_level_db", "out_level_db", "gain_reduction_db"],
                [(a, b, c) for a, b, c in zip(in_db, out_db, gr_db)],
            )
            arts.append(Artifact(kind="csv", path=str(cp), description="Compressor curve"))

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(
            feature="compressor",
            mode="loopback_realtime",
            summary={"fs": sr, "thr_db": thr_db, "ratio": ratio, "gain_offset_db": gain_offset_db, "no_compression": bool(ratio <= 1.05)},
            artifacts=arts,
        )

    return _start_async("compressor_loopback", impl, stop_event)


# ============================================================
# Attack/Release loopback (async)
# ============================================================

def start_attack_release_loopback(
    request: ARLoopbackRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> Handle:
    stop_event = stop_event or threading.Event()

    def impl() -> Result:
        sd = _import_sounddevice()
        sr = int(request.sample_rate or 48000)
        ch = int(request.channels or 1)

        n = int(sr * float(request.duration_s))
        tt = np.arange(n) / sr
        step_pt = n // 2

        env = np.ones(n, dtype=np.float32) * (float(request.amp) * 0.2)
        env[step_pt:] = float(request.amp)
        sig = (env * np.sin(2 * np.pi * float(request.freq_hz) * tt)).astype(np.float32)
        playback = sig[:, None] if ch == 1 else np.tile(sig[:, None], (1, ch))

        run_id = _new_run_id()
        ts = _utc_now_iso()
        outdir = _resolve_out_dir(request.out_dir, run_id)
        _ensure_dir(outdir)

        def payload(buf: np.ndarray, chunk: int) -> dict[str, Any]:
            env_db, t_env = _rms_envelope_db(buf, sr, float(request.rms_win_ms))
            return {"time_s": t_env[-200:], "envelope_db": env_db[-200:]}

        _emit_log(on_log, "INFO", "A/R loopback started", input_device=request.input_device, output_device=request.output_device)
        _emit_progress(on_progress, "stimulus", message="stimulus built", meta={"chunk": 0, "data": {}})

        runner = _DuplexRunner(
            sd,
            playback,
            sr=sr,
            in_ch=ch,
            out_ch=ch,
            input_device=request.input_device,
            output_device=request.output_device,
            blocksize=request.blocksize,
            stop_event=stop_event,
            on_progress=on_progress,
            on_log=on_log,
            make_payload=payload,
        )
        rx = runner.run()
        mono = _as_mono(rx)

        _emit_progress(on_progress, "analyze", message="estimate attack/release", meta={"chunk": 0, "data": {}})
        env_db, t_env = _rms_envelope_db(mono, sr, float(request.rms_win_ms))
        atk, rel = _estimate_attack_release(env_db, t_env)

        arts: list[Artifact] = []
        if request.export_wav:
            wp = outdir / "ar_recorded.wav"
            _emit_progress(on_progress, "export", message="export wav", meta={"chunk": 0, "data": {}})
            _wav_write(wp, rx, sr)
            arts.append(Artifact(kind="wav", path=str(wp), description="Recorded A/R capture"))

        if request.export_csv:
            cp = outdir / "ar_envelope.csv"
            _emit_progress(on_progress, "export", message="export csv", meta={"chunk": 0, "data": {}})
            _write_csv_with_meta(
                cp,
                {"feature": "attack_release", "mode": "loopback_realtime", "run_id": run_id, "timestamp": ts, "sample_rate": sr, "attack_ms": atk, "release_ms": rel, "rms_win_ms": request.rms_win_ms},
                ["time_s", "envelope_db"],
                [(a, b) for a, b in zip(t_env, env_db)],
            )
            arts.append(Artifact(kind="csv", path=str(cp), description="A/R envelope"))

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(
            feature="attack_release",
            mode="loopback_realtime",
            summary={"fs": sr, "attack_ms": atk, "release_ms": rel, "rms_win_ms": float(request.rms_win_ms)},
            artifacts=arts,
        )

    return _start_async("attack_release_loopback", impl, stop_event)


# ============================================================
# Export list (safe for GUI imports)
# ============================================================

__all__ = [
    "BackendError",
    "InvalidRequestError",
    "DeviceError",
    "AudioIOError",
    "DSPError",
    "CancelledError",
    "LogEvent",
    "ProgressEvent",
    "Artifact",
    "Result",
    "Handle",
    "ListDevicesRequest",
    "ReadWavRequest",
    "ThdOfflineRequest",
    "CompressorOfflineRequest",
    "AROfflineRequest",
    "CompareRequest",
    "ThdLoopbackRequest",
    "CompressorLoopbackRequest",
    "ARLoopbackRequest",
    "LoopbackRecordRequest",
    "run_list_devices",
    "run_wav_read",
    "run_thd_offline",
    "run_compressor_offline",
    "run_attack_release_offline",
    "run_compare",
    "start_thd_loopback",
    "start_compressor_loopback",
    "start_attack_release_loopback",
    "start_loopback_record",
]
