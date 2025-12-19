# analysis/offline.py
from __future__ import annotations

# Allow running this module directly (python analysis/offline.py) without package errors.
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
from pathlib import Path
import numpy as np

from analysis.common import (
    BackendError, InvalidRequestError, DSPError,
    Result, Artifact,
    ListDevicesRequest, ReadWavRequest,
    ThdOfflineRequest, CompressorOfflineRequest, AROfflineRequest, CompareRequest,
    _import_sounddevice,
    _wav_read, _as_mono,
    _compute_thd, _rms_envelope_db, _estimate_attack_release,
    _new_run_id, _utc_now_iso, _resolve_out_dir, _ensure_dir, _write_csv_with_meta,
)

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

def run_wav_read(request: ReadWavRequest) -> Result:
    p = Path(request.path)
    if not p.is_file():
        raise InvalidRequestError(f"WAV not found: {p}")
    sr, x = _wav_read(p)
    ch = 1 if x.ndim == 1 else int(x.shape[1])
    return Result(feature="audio_io", mode="offline", summary={"path": str(p), "fs": sr, "n_channels": ch, "n_samples": int(x.shape[0])})

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
            {
                "schema": "audio_toolkit_csv_v1",
                "feature": "thd",
                "mode": "offline",
                "run_id": run_id,
                "timestamp_utc": ts,
                "sample_rate": sr,
                "channels": 1 if x.ndim == 1 else int(x.shape[1]),
                "source_wav": str(p),
                "freq_hz": request.freq_hz,
                "hmax": request.hmax,
                "thd_ratio": float(thd),
                "thd_percent": float(100.0 * thd),
                "thd_db": float(20.0 * math.log10(thd + 1e-15)),
            },
            ["harmonic", "freq_hz", "mag_db"],
            [(i + 1, fk, dbv) for i, (fk, dbv) in enumerate(harms)],
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="THD harmonics", meta={"schema":"audio_toolkit_csv_v1","columns":["harmonic","freq_hz","mag_db"]}))

    return Result(
        feature="thd",
        mode="offline",
        summary={"fs": sr, "freq_hz": float(request.freq_hz), "hmax": int(request.hmax), "thd_ratio": float(thd), "thd_percent": float(100.0 * thd), "thd_db": float(20.0 * math.log10(thd + 1e-15))},
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
    ylv = xlv.copy()
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
            {
                "schema": "audio_toolkit_csv_v1",
                "feature": "compressor",
                "mode": "offline",
                "run_id": run_id,
                "timestamp_utc": ts,
                "sample_rate": sr,
                "channels": 1 if x.ndim == 1 else int(x.shape[1]),
                "source_wav": str(p),
                "freq_hz": request.freq_hz,
                "thr_db": thr_db,
                "ratio": ratio,
                "gain_offset_db": gain_offset_db,
            },
            ["in_level_db", "out_level_db"],
            list(zip(xlv.astype(float), ylv.astype(float))),
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="Compressor curve (heuristic)", meta={"schema":"audio_toolkit_csv_v1","columns":["in_level_db","out_level_db"]}))

    return Result(feature="compressor", mode="offline", summary={"fs": sr, "thr_db": thr_db, "ratio": ratio, "gain_offset_db": gain_offset_db, "no_compression": bool(ratio <= 1.05)}, artifacts=arts)

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
            {
                "schema": "audio_toolkit_csv_v1",
                "feature": "attack_release",
                "mode": "offline",
                "run_id": run_id,
                "timestamp_utc": ts,
                "sample_rate": sr,
                "channels": 1 if x.ndim == 1 else int(x.shape[1]),
                "source_wav": str(p),
                "rms_win_ms": request.rms_win_ms,
                "attack_ms": atk,
                "release_ms": rel,
            },
            ["time_s", "envelope_db"],
            list(zip(tt, env)),
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="A/R envelope", meta={"schema":"audio_toolkit_csv_v1","columns":["time_s","envelope_db"]}))

    return Result(feature="attack_release", mode="offline", summary={"fs": sr, "rms_win_ms": float(request.rms_win_ms), "attack_ms": atk, "release_ms": rel}, artifacts=arts)

# compare functions (same as before)
import numpy as _np
def _estimate_lag(x: _np.ndarray, y: _np.ndarray, max_lag: int) -> int:
    x = x - _np.mean(x); y = y - _np.mean(y)
    n = min(len(x), len(y)); x = x[:n]; y = y[:n]
    nfft = 1 << int(math.ceil(math.log2(max(8, 2 * n))))
    X = _np.fft.rfft(x, nfft); Y = _np.fft.rfft(y, nfft)
    c = _np.fft.irfft(X * _np.conj(Y), nfft)
    c = _np.concatenate([c[-(n - 1) :], c[:n]])
    mid = len(c) // 2
    lo = max(0, mid - max_lag); hi = min(len(c), mid + max_lag + 1)
    seg = c[lo:hi]
    return int(_np.argmax(seg) - (mid - lo))

def _apply_lag(ref: _np.ndarray, out: _np.ndarray, lag: int) -> tuple[_np.ndarray, _np.ndarray]:
    if lag > 0:
        ref2 = ref[:-lag]; out2 = out[lag:]
    elif lag < 0:
        l = -lag; ref2 = ref[l:]; out2 = out[:-l]
    else:
        ref2, out2 = ref, out
    n = min(len(ref2), len(out2))
    return ref2[:n], out2[:n]

def _estimate_gain(ref: _np.ndarray, out: _np.ndarray) -> float:
    denom = float(_np.dot(out, out) + 1e-12)
    return float(_np.dot(ref, out) / denom)

def run_compare(request: CompareRequest) -> Result:
    pr = Path(request.ref_wav_path); po = Path(request.tgt_wav_path)
    if not pr.is_file() or not po.is_file():
        raise InvalidRequestError("ref_wav_path/tgt_wav_path not found")

    sr_r, r = _wav_read(pr); sr_o, o = _wav_read(po)
    if sr_r != sr_o:
        raise InvalidRequestError("Sample rate mismatch")

    r = _as_mono(r); o = _as_mono(o)
    max_lag = int(max(1, float(request.max_lag_s) * sr_r))
    lag = _estimate_lag(r, o, max_lag)
    rr, oo = _apply_lag(r, o, lag)
    g = _estimate_gain(rr, oo)
    oo_g = oo / (g if g != 0 else 1.0)
    res = oo_g - rr

    rms_r = float(_np.sqrt(_np.mean(rr * rr) + 1e-12))
    rms_res = float(_np.sqrt(_np.mean(res * res) + 1e-12))
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
            {
                "schema": "audio_toolkit_csv_v1",
                "feature": "compare",
                "mode": "offline",
                "run_id": run_id,
                "timestamp_utc": ts,
                "sample_rate": sr_r,
                "channels": 1,
                "ref_wav_path": str(pr),
                "tgt_wav_path": str(po),
                "max_lag_s": request.max_lag_s,
                "lag_samples": lag,
                "gain_linear": g,
            },
            ["metric", "value"],
            [("lag_samples", lag), ("lag_ms", 1000.0 * lag / sr_r), ("gain_error_db", gain_err_db), ("snr_db", snr_db)],
        )
        arts.append(Artifact(kind="csv", path=str(cp), description="Compare metrics", meta={"schema":"audio_toolkit_csv_v1","columns":["metric","value"]}))

    return Result(feature="compare", mode="offline", summary={"fs": sr_r, "lag_samples": lag, "lag_ms": 1000.0 * lag / sr_r, "gain_error_db": gain_err_db, "snr_db": snr_db}, artifacts=arts)
