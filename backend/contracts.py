# backend/contracts.py
"""
Backend API Contract for Audio Measurement Toolkit.

NON-NEGOTIABLE:
- GUI_D_3_2_1.py is the immutable reference GUI.
- Backend must NOT import tkinter / messagebox / depend on GUI state.
- GUI should call ONLY the public facade functions defined in this module.

Public API rules:
- Sync (fast):   run_xxx(request) -> XxxResult
- Async (long):  start_xxx(request, *, stop_event, on_progress, on_log) -> XxxHandle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol, Sequence
import threading
import time
import os

# =========================
# Error model (standard)
# =========================

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


# =========================
# Common data structures
# =========================

@dataclass(frozen=True)
class PlotSpec:
    """
    Plot description ONLY (no matplotlib figure objects).
    GUI-side code (or utils/plot_windows.py) may translate PlotSpec -> real plots.
    """
    kind: Literal[
        "thd_snapshot",
        "compressor_curve",
        "ar_envelope",
        "compare_overlay",
        "spectrum",
    ]
    title: str
    x: list[float] | None = None
    y: list[float] | None = None
    series: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    """File outputs produced by backend."""
    kind: Literal["wav", "csv", "json", "txt", "other"]
    path: str
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEvent:
    """Streaming progress for long tasks."""
    phase: str
    percent: float | None = None  # 0..100
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LogEvent:
    """Streaming log event for GUI to display."""
    level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseResult:
    """
    Standard result payload.
    GUI can display summary keys and list artifacts/plots.
    """
    summary: dict[str, Any] = field(default_factory=dict)
    plots: list[PlotSpec] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


# =========================
# Handle interface
# =========================

class XxxHandle(Protocol):
    def join(self, timeout: float | None = None) -> BaseResult: ...
    def cancel(self) -> None: ...
    def is_running(self) -> bool: ...


@dataclass
class ThreadHandle:
    """
    Simple worker-thread handle implementing:
    - join(): returns BaseResult (or raises)
    - cancel(): cooperative cancellation via stop_event
    - is_running()
    """
    _thread: threading.Thread
    _stop_event: threading.Event
    _result_box: dict[str, Any]
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def join(self, timeout: float | None = None) -> BaseResult:
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("Worker still running.")
        with self._lock:
            err = self._result_box.get("error")
            if err is not None:
                raise err
            res = self._result_box.get("result")
            if res is None:
                raise BackendError("Worker finished without result.")
            return res

    def cancel(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread.is_alive()


# =========================
# Requests / Results
# =========================

# ---- Audio I/O ----

@dataclass(frozen=True)
class ListDevicesRequest:
    """List available audio devices."""
    pass


@dataclass
class ListDevicesResult(BaseResult):
    pass


@dataclass(frozen=True)
class ValidateDeviceRequest:
    """Validate indices and basic stream parameters."""
    input_device: int | None
    output_device: int | None
    samplerate: float | None = None
    input_channels: int = 1
    output_channels: int = 2


@dataclass
class ValidateDeviceResult(BaseResult):
    pass


@dataclass(frozen=True)
class ReadWavRequest:
    path: str
    mono: bool = True


@dataclass
class ReadWavResult(BaseResult):
    fs: int | None = None
    n_samples: int | None = None
    n_channels: int | None = None


@dataclass(frozen=True)
class WriteWavRequest:
    path: str
    fs: int
    samples: Sequence[float] | Sequence[Sequence[float]]  # mono or multi


@dataclass
class WriteWavResult(BaseResult):
    pass


@dataclass(frozen=True)
class LoopbackRecordRequest:
    """
    Play input_wav_path to output_device and record from input_device,
    then write recorded to output_path.
    """
    input_wav_path: str
    input_device: int | None
    output_device: int | None
    output_path: str
    input_channels: int = 1


@dataclass
class LoopbackRecordResult(BaseResult):
    pass


# ---- Compare ----

@dataclass(frozen=True)
class CompareRequest:
    ref_wav_path: str
    tgt_wav_path: str
    freq_hz: float = 1000.0
    hmax: int = 5
    max_lag_seconds: float = 5.0


@dataclass
class CompareResult(BaseResult):
    pass


# ---- THD ----

@dataclass(frozen=True)
class ThdOfflineRequest:
    wav_path: str
    freq_hz: float = 1000.0
    hmax: int = 5


@dataclass
class ThdResult(BaseResult):
    pass


@dataclass(frozen=True)
class ThdRealtimeRequest:
    freq_hz: float = 1000.0
    amp: float = 0.7
    hmax: int = 5
    input_device: int | None = None
    output_device: int | None = None
    samplerate: float | None = None
    base_dir: str = "."


# ---- Compressor ----

@dataclass(frozen=True)
class CompressorOfflineRequest:
    wav_path: str
    freq_hz: float = 1000.0


@dataclass
class CompressorResult(BaseResult):
    pass


@dataclass(frozen=True)
class CompressorRealtimeRequest:
    freq_hz: float = 1000.0
    amp_max: float = 1.36
    input_device: int | None = None
    output_device: int | None = None
    samplerate: float | None = None
    base_dir: str = "."


# ---- Attack/Release ----

@dataclass(frozen=True)
class AROfflineRequest:
    wav_path: str
    rms_win_ms: float = 5.0


@dataclass
class ARResult(BaseResult):
    pass


@dataclass(frozen=True)
class ARRealtimeRequest:
    freq_hz: float = 1000.0
    amp: float = 0.7
    rms_win_ms: float = 5.0
    input_device: int | None = None
    output_device: int | None = None
    samplerate: float | None = None
    base_dir: str = "."


# =========================
# Internal helpers
# =========================

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_file(path: str) -> None:
    if not path or not os.path.isfile(path):
        raise InvalidRequestError(f"File not found: {path}")


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise BackendError("numpy is required for backend contracts.") from e
    return np


def _to_1d_float_array(x: Any):
    np = _import_numpy()
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2:
        # take first channel for mono
        arr = arr[:, 0]
    return arr.astype(float)


def _safe_log(on_log: Callable[[LogEvent], None] | None, level: str, msg: str, **meta: Any) -> None:
    if on_log:
        on_log(LogEvent(level=level, message=msg, meta=dict(meta)))


def _safe_progress(on_progress: Callable[[ProgressEvent], None] | None, phase: str, percent: float | None, msg: str, **meta: Any) -> None:
    if on_progress:
        on_progress(ProgressEvent(phase=phase, percent=percent, message=msg, meta=dict(meta)))


def _make_handle(
    worker: Callable[[], BaseResult],
    *,
    name: str,
    stop_event: threading.Event,
) -> ThreadHandle:
    box: dict[str, Any] = {}
    lock = threading.Lock()

    def run():
        try:
            if stop_event.is_set():
                raise CancelledError()
            res = worker()
            with lock:
                box["result"] = res
        except Exception as e:
            with lock:
                box["error"] = e

    t = threading.Thread(target=run, daemon=True, name=name)
    t.start()
    return ThreadHandle(t, stop_event, box, lock)


def _normalize_align_result(ret: Any):
    """
    Accept several possible signatures from analysis.compare.align_signals:
    - (a_aligned, b_aligned, lag)
    - (a_aligned, b_aligned, lag, extra...)
    """
    if not isinstance(ret, (tuple, list)) or len(ret) < 3:
        raise DSPError("align_signals() returned unexpected value.")
    a_al, b_al, lag = ret[0], ret[1], ret[2]
    return a_al, b_al, lag


def _normalize_gain_match_result(ret: Any):
    """
    Accept possible signatures from analysis.compare.gain_match:
    - (b_matched, gain_error_db)
    - (b_matched, gain_error_db, extra...)
    """
    if not isinstance(ret, (tuple, list)) or len(ret) < 2:
        raise DSPError("gain_match() returned unexpected value.")
    return ret[0], ret[1]


def _try_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call function, raising DSPError with context."""
    try:
        return fn(*args, **kwargs)
    except CancelledError:
        raise
    except Exception as e:
        raise DSPError(f"{fn.__module__}.{fn.__name__} failed: {e}") from e


# =========================
# Public Facade API
# =========================

# -------- Audio device APIs --------

def run_list_devices(request: ListDevicesRequest) -> ListDevicesResult:
    """
    List available audio input/output devices.

    Returns:
        ListDevicesResult.summary:
            - inputs: list[str]   (GUI-friendly device strings)
            - outputs: list[str]
            - signature: str|None (optional, for GUI caching if available)
    """
    from audio import devices as dev  # project module

    inputs, outputs = dev.list_devices(raise_on_error=True)
    signature = None
    if hasattr(dev, "get_devices_signature"):
        try:
            signature = dev.get_devices_signature()
        except Exception:
            signature = None

    res = ListDevicesResult()
    res.summary = {"inputs": inputs, "outputs": outputs, "signature": signature}
    return res


def run_validate_device(request: ValidateDeviceRequest) -> ValidateDeviceResult:
    """
    Validate a device configuration for play/record.

    Backend strategy:
    - If audio.devices.validate_device_config exists, use it.
    - Else, perform a conservative validation using sounddevice query if available.
    """
    # Prefer project helper
    from audio import devices as dev

    if hasattr(dev, "validate_device_config"):
        ok, details = dev.validate_device_config(
            request.input_device,
            request.output_device,
            samplerate=request.samplerate,
            input_channels=request.input_channels,
            output_channels=request.output_channels,
        )
        if not ok:
            raise DeviceError(details.get("reason", "Invalid device config"))
        res = ValidateDeviceResult()
        res.summary = {"ok": True, "details": details}
        return res

    # Fallback: sounddevice query (no GUI dependency)
    try:
        import sounddevice as sd  # type: ignore
        details: dict[str, Any] = {}
        if request.input_device is not None:
            d_in = sd.query_devices(request.input_device)
            details["input"] = d_in
            if request.input_channels > int(d_in.get("max_input_channels", 0)):
                raise DeviceError("Input channels exceed device capability")
        if request.output_device is not None:
            d_out = sd.query_devices(request.output_device)
            details["output"] = d_out
            if request.output_channels > int(d_out.get("max_output_channels", 0)):
                raise DeviceError("Output channels exceed device capability")
        sr = request.samplerate
        if sr is None:
            # Try using default samplerate from output device if present
            try:
                sr = float(details.get("output", {}).get("default_samplerate"))
            except Exception:
                sr = None
        res = ValidateDeviceResult()
        res.summary = {"ok": True, "details": details, "samplerate": sr}
        return res
    except DeviceError:
        raise
    except Exception as e:
        raise DeviceError(f"Device validation failed: {e}") from e


# -------- WAV APIs --------

def run_wav_read(request: ReadWavRequest) -> ReadWavResult:
    """
    Read WAV file. Returns metadata in summary.

    Note: Result does not return full samples by default (GUI doesn't need them).
    """
    _ensure_file(request.path)
    from audio import wav_io

    fs, data = wav_io.read_wav(request.path)
    if fs is None:
        raise AudioIOError(f"Cannot read wav: {request.path}")

    np = _import_numpy()
    arr = np.asarray(data)
    n_channels = 1 if arr.ndim == 1 else arr.shape[1]
    n_samples = arr.shape[0]

    res = ReadWavResult()
    res.fs = int(fs)
    res.n_samples = int(n_samples)
    res.n_channels = int(n_channels)
    res.summary = {"path": request.path, "fs": int(fs), "n_samples": int(n_samples), "n_channels": int(n_channels)}
    return res


def run_wav_write(request: WriteWavRequest) -> WriteWavResult:
    """
    Write WAV file using project wav_io.write_wav if available.

    If wav_io.write_wav has different signature, this tries common patterns.
    """
    from audio import wav_io
    # attempt common signatures: write_wav(path, data, fs) or write_wav(path, data, fs, ...)
    ok = None
    try:
        ok = wav_io.write_wav(request.path, request.samples, request.fs)
    except TypeError:
        try:
            ok = wav_io.write_wav(request.path, request.fs, request.samples)
        except Exception as e:
            raise AudioIOError(f"Cannot write wav: {e}") from e
    except Exception as e:
        raise AudioIOError(f"Cannot write wav: {e}") from e

    if ok is False:
        raise AudioIOError(f"write_wav returned False: {request.path}")

    res = WriteWavResult()
    res.artifacts.append(Artifact(kind="wav", path=request.path, description="WAV written"))
    res.summary = {"path": request.path, "fs": int(request.fs)}
    return res


# -------- Compare APIs --------

def run_compare(request: CompareRequest) -> CompareResult:
    """
    Align & gain-match two WAV signals and compute residual metrics.

    Returns:
        summary:
          - fs
          - latency_samples, latency_ms
          - gain_error_db
          - metrics (dict)
        plots:
          - compare_overlay (series ref/tgt)
    """
    _ensure_file(request.ref_wav_path)
    _ensure_file(request.tgt_wav_path)

    from audio import wav_io
    from analysis import compare as cmp

    fs1, a = wav_io.read_wav(request.ref_wav_path)
    fs2, b = wav_io.read_wav(request.tgt_wav_path)
    if fs1 is None or fs2 is None:
        raise AudioIOError("Cannot read one or both wav files")
    if int(fs1) != int(fs2):
        raise InvalidRequestError(f"Fs mismatch: {fs1} vs {fs2}")

    np = _import_numpy()
    a = _to_1d_float_array(a)
    b = _to_1d_float_array(b)

    max_lag = int(float(fs1) * float(request.max_lag_seconds))
    a_al, b_al, lag = _normalize_align_result(_try_call(cmp.align_signals, a, b, max_lag_samples=max_lag))
    b_gm, gain_err = _normalize_gain_match_result(_try_call(cmp.gain_match, a_al, b_al))
    metrics = _try_call(cmp.residual_metrics, a_al, b_gm, int(fs1), float(request.freq_hz), int(request.hmax))

    # overlay plot (downsample if huge)
    def _downsample(x: Any, max_points: int = 50000):
        arr = np.asarray(x)
        n = arr.shape[0]
        if n <= max_points:
            return arr
        step = max(1, n // max_points)
        return arr[::step]

    a_p = _downsample(a_al)
    b_p = _downsample(b_gm)

    res = CompareResult()
    res.summary = {
        "fs": int(fs1),
        "latency_samples": int(lag),
        "latency_ms": float(int(lag) / float(fs1) * 1000.0),
        "gain_error_db": float(gain_err),
        "metrics": metrics,
    }
    res.plots.append(
        PlotSpec(
            kind="compare_overlay",
            title="Compare overlay (aligned)",
            series=[
                {"label": "ref", "x": list(range(len(a_p))), "y": a_p.astype(float).tolist()},
                {"label": "tgt", "x": list(range(len(b_p))), "y": b_p.astype(float).tolist()},
            ],
            meta={"fs": int(fs1), "downsampled": True},
        )
    )
    return res


# -------- THD APIs --------

def run_thd_offline(request: ThdOfflineRequest) -> ThdResult:
    """
    Compute THD metrics from a WAV file (offline).

    Returns:
      summary: thd_percent/thd_db + harmonics if available.
      plots: thd_snapshot (data may be inside summary/meta)
    """
    _ensure_file(request.wav_path)
    from audio import wav_io
    from analysis import thd as thd_mod

    fs, sig = wav_io.read_wav(request.wav_path)
    if fs is None:
        raise AudioIOError(f"Cannot read wav: {request.wav_path}")
    sig = _to_1d_float_array(sig)

    out = _try_call(thd_mod.compute_thd, sig, int(fs), float(request.freq_hz), int(request.hmax))
    res = ThdResult()
    res.summary = {
        "path": request.wav_path,
        "fs": int(fs),
        "freq_hz": float(request.freq_hz),
        "hmax": int(request.hmax),
        "thd_percent": out.get("thd_percent_manual", out.get("thd_percent")),
        "thd_db": out.get("thd_db_manual", out.get("thd_db")),
        "harmonics": out.get("harmonics_manual", out.get("harmonics", {})),
        "raw": out,  # keep for debugging/advanced UI if needed
    }
    res.plots.append(
        PlotSpec(
            kind="thd_snapshot",
            title="THD snapshot",
            meta={"fs": int(fs), "freq_hz": float(request.freq_hz), "hmax": int(request.hmax)},
        )
    )
    return res


def start_thd_realtime(
    request: ThdRealtimeRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> ThreadHandle:
    """
    Realtime THD measurement (loopback via soundcard).

    Worker steps:
    - choose samplerate
    - generate stimulus tone (analysis.live_measurements.generate_thd_tone)
    - play/record (audio.playrec.play_and_record)
    - analyze (analysis.live_measurements.analyze_thd_capture)
    - export (csv row + wav artifacts if supported by live_measurements)
    """
    stop_event = stop_event or threading.Event()

    def worker() -> BaseResult:
        from audio import devices as dev
        from audio import playrec
        from analysis import live_measurements

        _safe_progress(on_progress, "validate", 0, "Validate request")
        if request.freq_hz <= 0:
            raise InvalidRequestError("freq_hz must be > 0")
        if request.hmax < 1:
            raise InvalidRequestError("hmax must be >= 1")
        if stop_event.is_set():
            raise CancelledError()

        try:
            fs = int(request.samplerate or dev.default_samplerate(request.output_device or None))
        except Exception:
            fs = int(request.samplerate or 48000)

        _safe_progress(on_progress, "generate", 10, "Generate THD tone", fs=fs)
        tone = _try_call(live_measurements.generate_thd_tone, float(request.freq_hz), float(request.amp), fs)

        if stop_event.is_set():
            raise CancelledError()

        _safe_progress(on_progress, "playrec", 30, "Play & record")
        _safe_log(on_log, "INFO", f"Play/rec fs={fs}, f={request.freq_hz}Hz amp={request.amp}")
        recorded = playrec.play_and_record(
            tone,
            fs,
            request.input_device,
            request.output_device,
            stop_event,
            log=(lambda m: _safe_log(on_log, "INFO", str(m))),
            input_channels=1,
        )
        if recorded is None or len(recorded) == 0:
            raise AudioIOError("No recorded data")

        if stop_event.is_set():
            raise CancelledError()

        _safe_progress(on_progress, "analyze", 70, "Analyze THD")
        out = _try_call(live_measurements.analyze_thd_capture, recorded, fs, float(request.freq_hz), int(request.hmax))

        res = ThdResult()
        res.summary = {
            "fs": fs,
            "freq_hz": float(request.freq_hz),
            "hmax": int(request.hmax),
            "thd_percent": out.get("thd_percent_manual", out.get("thd_percent")),
            "thd_db": out.get("thd_db_manual", out.get("thd_db")),
            "harmonics": out.get("harmonics_manual", out.get("harmonics", {})),
            "raw": out,
        }

        # Export artifacts if helpers exist
        _safe_progress(on_progress, "export", 90, "Export artifacts")
        base_dir = request.base_dir or "."
        try:
            csv_path = _try_call(
                live_measurements.append_csv_row,
                (_now_ts(), "THD",
                 f"{res.summary.get('thd_percent', 0.0):.6f}%",
                 f"{res.summary.get('thd_db', 0.0):.3f} dB"),
                base_dir,
            )
            res.artifacts.append(Artifact(kind="csv", path=str(csv_path), description="Measurement log (CSV)"))
        except Exception:
            # CSV export is optional; don't fail run for this
            pass

        try:
            arts = _try_call(live_measurements.save_artifacts, "thd", tone, recorded, fs, base_dir)
            if isinstance(arts, dict):
                if "tx" in arts:
                    res.artifacts.append(Artifact(kind="wav", path=str(arts["tx"]), description="TX tone"))
                if "rx" in arts:
                    res.artifacts.append(Artifact(kind="wav", path=str(arts["rx"]), description="RX capture"))
        except Exception:
            pass

        res.plots.append(
            PlotSpec(
                kind="thd_snapshot",
                title="THD snapshot",
                meta={"fs": fs, "freq_hz": float(request.freq_hz), "hmax": int(request.hmax)},
            )
        )
        _safe_progress(on_progress, "done", 100, "Done")
        return res

    return _make_handle(worker, name="thd_realtime", stop_event=stop_event)


# -------- Compressor APIs --------

def run_compressor_offline(request: CompressorOfflineRequest) -> CompressorResult:
    """
    Offline compressor analysis from a WAV containing stepped tone response.
    """
    _ensure_file(request.wav_path)
    from audio import wav_io
    from analysis import compressor as comp

    fs, sig = wav_io.read_wav(request.wav_path)
    if fs is None:
        raise AudioIOError(f"Cannot read wav: {request.wav_path}")
    sig = _to_1d_float_array(sig)

    tone_info = _try_call(comp.build_stepped_tone, float(request.freq_hz), int(fs))
    meta = tone_info.get("meta", tone_info)  # tolerate different return shapes
    curve = _try_call(comp.compression_curve, sig, meta, int(fs), float(request.freq_hz))

    res = CompressorResult()
    res.summary = {
        "path": request.wav_path,
        "fs": int(fs),
        "freq_hz": float(request.freq_hz),
        "no_compression": curve.get("no_compression"),
        "thr_db": curve.get("thr_db"),
        "ratio": curve.get("ratio"),
        "gain_offset_db": curve.get("gain_offset_db"),
        "raw": curve,
    }
    res.plots.append(
        PlotSpec(
            kind="compressor_curve",
            title="Compressor curve",
            meta={"fs": int(fs), "freq_hz": float(request.freq_hz)},
        )
    )
    return res


def start_compressor_realtime(
    request: CompressorRealtimeRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> ThreadHandle:
    """
    Realtime compressor measurement via loopback.
    Uses:
      - analysis.live_measurements.generate_compressor_tone
      - audio.playrec.play_and_record
      - analysis.live_measurements.analyze_compressor_capture
      - analysis.live_measurements.append_csv_row/save_artifacts (optional)
    """
    stop_event = stop_event or threading.Event()

    def worker() -> BaseResult:
        from audio import devices as dev
        from audio import playrec
        from analysis import live_measurements

        if request.freq_hz <= 0:
            raise InvalidRequestError("freq_hz must be > 0")
        if request.amp_max <= 0:
            raise InvalidRequestError("amp_max must be > 0")
        if stop_event.is_set():
            raise CancelledError()

        try:
            fs = int(request.samplerate or dev.default_samplerate(request.output_device or None))
        except Exception:
            fs = int(request.samplerate or 48000)

        _safe_progress(on_progress, "generate", 10, "Generate compressor stepped tone", fs=fs)
        tone, meta = _try_call(live_measurements.generate_compressor_tone, float(request.freq_hz), fs, float(request.amp_max))

        if stop_event.is_set():
            raise CancelledError()

        _safe_progress(on_progress, "playrec", 30, "Play & record")
        _safe_log(on_log, "INFO", f"Play/rec fs={fs}, f={request.freq_hz}Hz amp_max={request.amp_max}")
        recorded = playrec.play_and_record(
            tone,
            fs,
            request.input_device,
            request.output_device,
            stop_event,
            log=(lambda m: _safe_log(on_log, "INFO", str(m))),
            input_channels=1,
        )
        if recorded is None or len(recorded) == 0:
            raise AudioIOError("No recorded data")

        if stop_event.is_set():
            raise CancelledError()

        _safe_progress(on_progress, "analyze", 70, "Analyze compressor capture")
        curve = _try_call(live_measurements.analyze_compressor_capture, recorded, meta, fs)

        res = CompressorResult()
        res.summary = {
            "fs": fs,
            "freq_hz": float(request.freq_hz),
            "amp_max": float(request.amp_max),
            "no_compression": curve.get("no_compression"),
            "thr_db": curve.get("thr_db"),
            "ratio": curve.get("ratio"),
            "gain_offset_db": curve.get("gain_offset_db"),
            "raw": curve,
        }

        _safe_progress(on_progress, "export", 90, "Export artifacts")
        base_dir = request.base_dir or "."
        try:
            csv_path = _try_call(
                live_measurements.append_csv_row,
                (_now_ts(), "COMPRESSOR",
                 f"thr={res.summary.get('thr_db')}",
                 f"ratio={res.summary.get('ratio')}"),
                base_dir,
            )
            res.artifacts.append(Artifact(kind="csv", path=str(csv_path), description="Measurement log (CSV)"))
        except Exception:
            pass

        try:
            arts = _try_call(live_measurements.save_artifacts, "compressor", tone, recorded, fs, base_dir)
            if isinstance(arts, dict):
                if "tx" in arts:
                    res.artifacts.append(Artifact(kind="wav", path=str(arts["tx"]), description="TX tone"))
                if "rx" in arts:
                    res.artifacts.append(Artifact(kind="wav", path=str(arts["rx"]), description="RX capture"))
        except Exception:
            pass

        res.plots.append(
            PlotSpec(
                kind="compressor_curve",
                title="Compressor curve",
                meta={"fs": fs, "freq_hz": float(request.freq_hz)},
            )
        )
        _safe_progress(on_progress, "done", 100, "Done")
        return res

    return _make_handle(worker, name="compressor_realtime", stop_event=stop_event)


# -------- Attack/Release APIs --------

def run_ar_offline(request: AROfflineRequest) -> ARResult:
    """
    Offline attack/release time measurement from a WAV file.
    """
    _ensure_file(request.wav_path)
    from audio import wav_io
    from analysis import attack_release as ar

    fs, sig = wav_io.read_wav(request.wav_path)
    if fs is None:
        raise AudioIOError(f"Cannot read wav: {request.wav_path}")
    sig = _to_1d_float_array(sig)

    times_dict = _try_call(ar.attack_release_times, sig, int(fs), float(request.rms_win_ms))
    res = ARResult()
    res.summary = {"path": request.wav_path, "fs": int(fs), "rms_win_ms": float(request.rms_win_ms), **(times_dict or {})}
    res.plots.append(
        PlotSpec(
            kind="ar_envelope",
            title="Attack/Release envelope",
            meta={"fs": int(fs), "rms_win_ms": float(request.rms_win_ms)},
        )
    )
    return res


def start_attack_release_realtime(
    request: ARRealtimeRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> ThreadHandle:
    """
    Realtime A/R measurement via loopback.
    Uses:
      - analysis.attack_release.generate_step_tone
      - audio.playrec.play_and_record
      - analysis.attack_release.attack_release_times
    Optionally exports artifacts via analysis.live_measurements.save_artifacts (if available).
    """
    stop_event = stop_event or threading.Event()

    def worker() -> BaseResult:
        from audio import devices as dev
        from audio import playrec
        from analysis import attack_release as ar

        if request.freq_hz <= 0:
            raise InvalidRequestError("freq_hz must be > 0")
        if request.rms_win_ms <= 0:
            raise InvalidRequestError("rms_win_ms must be > 0")

        try:
            fs = int(request.samplerate or dev.default_samplerate(request.output_device or None))
        except Exception:
            fs = int(request.samplerate or 48000)

        _safe_progress(on_progress, "generate", 10, "Generate step tone", fs=fs)
        tone = _try_call(ar.generate_step_tone, float(request.freq_hz), fs, amp=float(request.amp))

        if stop_event.is_set():
            raise CancelledError()

        _safe_progress(on_progress, "playrec", 30, "Play & record")
        _safe_log(on_log, "INFO", f"Play/rec fs={fs}, f={request.freq_hz}Hz amp={request.amp}")
        recorded = playrec.play_and_record(
            tone,
            fs,
            request.input_device,
            request.output_device,
            stop_event,
            log=(lambda m: _safe_log(on_log, "INFO", str(m))),
            input_channels=1,
        )
        if recorded is None or len(recorded) == 0:
            raise AudioIOError("No recorded data")

        if stop_event.is_set():
            raise CancelledError()

        _safe_progress(on_progress, "analyze", 75, "Analyze envelope (attack/release)")
        times_dict = _try_call(ar.attack_release_times, _to_1d_float_array(recorded), fs, float(request.rms_win_ms))

        res = ARResult()
        res.summary = {
            "fs": fs,
            "freq_hz": float(request.freq_hz),
            "amp": float(request.amp),
            "rms_win_ms": float(request.rms_win_ms),
            **(times_dict or {}),
        }

        # Optional artifact export if live_measurements exists
        _safe_progress(on_progress, "export", 90, "Export artifacts (optional)")
        base_dir = request.base_dir or "."
        try:
            from analysis import live_measurements
            arts = _try_call(live_measurements.save_artifacts, "attack_release", tone, recorded, fs, base_dir)
            if isinstance(arts, dict):
                if "tx" in arts:
                    res.artifacts.append(Artifact(kind="wav", path=str(arts["tx"]), description="TX step tone"))
                if "rx" in arts:
                    res.artifacts.append(Artifact(kind="wav", path=str(arts["rx"]), description="RX capture"))
        except Exception:
            pass

        res.plots.append(
            PlotSpec(
                kind="ar_envelope",
                title="Attack/Release envelope",
                meta={"fs": fs, "rms_win_ms": float(request.rms_win_ms)},
            )
        )
        _safe_progress(on_progress, "done", 100, "Done")
        return res

    return _make_handle(worker, name="attack_release_realtime", stop_event=stop_event)


# -------- Loopback record API --------

def start_loopback_record(
    request: LoopbackRecordRequest,
    *,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    on_log: Callable[[LogEvent], None] | None = None,
) -> ThreadHandle:
    """
    Play input WAV, record loopback, write recorded WAV to output_path.
    """
    stop_event = stop_event or threading.Event()

    def worker() -> BaseResult:
        _ensure_file(request.input_wav_path)
        if not request.output_path:
            raise InvalidRequestError("output_path is required")

        from audio import wav_io
        from audio import playrec

        fs, sig = wav_io.read_wav(request.input_wav_path)
        if fs is None:
            raise AudioIOError("Cannot read input wav")

        sig = _to_1d_float_array(sig)

        _safe_progress(on_progress, "playrec", 25, "Play & record loopback", fs=int(fs))
        _safe_log(on_log, "INFO", f"Loopback play/rec fs={fs} file={request.input_wav_path}")
        recorded = playrec.play_and_record(
            sig,
            int(fs),
            request.input_device,
            request.output_device,
            stop_event,
            log=(lambda m: _safe_log(on_log, "INFO", str(m))),
            input_channels=int(request.input_channels),
        )
        if recorded is None or len(recorded) == 0:
            raise AudioIOError("No recorded data")

        if stop_event.is_set():
            raise CancelledError()

        _safe_progress(on_progress, "write", 80, "Write recorded wav")
        # Use project write_wav
        try:
            wav_io.write_wav(request.output_path, recorded, int(fs))
        except TypeError:
            # alternate signature
            wav_io.write_wav(request.output_path, int(fs), recorded)
        except Exception as e:
            raise AudioIOError(f"Cannot write recorded wav: {e}") from e

        res = LoopbackRecordResult()
        res.summary = {
            "input_wav_path": request.input_wav_path,
            "output_path": request.output_path,
            "fs": int(fs),
            "n_samples": int(len(recorded)),
        }
        res.artifacts.append(Artifact(kind="wav", path=request.output_path, description="Recorded loopback WAV"))
        _safe_progress(on_progress, "done", 100, "Done")
        return res

    return _make_handle(worker, name="loopback_record", stop_event=stop_event)
