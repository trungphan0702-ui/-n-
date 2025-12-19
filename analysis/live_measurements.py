# analysis/live_measurements.py
from __future__ import annotations

# Allow running this module directly (python analysis/live_measurements.py) without package errors.
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from analysis.common import (
    BackendError, InvalidRequestError, DeviceError, AudioIOError, DSPError, CancelledError,
    LogEvent, ProgressEvent,
    Artifact, Result, Handle,
    ThdLoopbackRequest, CompressorLoopbackRequest, ARLoopbackRequest, LoopbackRecordRequest,
    _start_async, _import_sounddevice,
    _new_run_id, _utc_now_iso, _resolve_out_dir, _ensure_dir,
    _wav_read, _wav_write, _as_mono,
    _spectrum_db, _compute_thd, _rms_envelope_db, _estimate_attack_release,
    _write_csv_with_meta, _emit_log, _emit_progress,
)

# ============================================================
# Realtime duplex runner (streams chunks)
# ============================================================

class _DuplexRunner:
    def __init__(
        self,
        sd,
        playback: np.ndarray,
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
        make_payload: Callable[[np.ndarray, int, float], dict[str, Any]] | None = None,
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
        started = time.time()

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
                t_sec = float(time.time() - started)
                payload = self.make_payload(buf, self._chunk, t_sec) if self.make_payload else {"t_sec": t_sec, "n_samples": int(buf.size)}
                payload.setdefault("t_sec", t_sec)
                payload.setdefault("n_samples", int(buf.size))
                payload.setdefault("sample_rate", int(self.sr))
                payload.setdefault("blocksize", int(self.blocksize))

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

def start_loopback_record(request: LoopbackRecordRequest, *, stop_event: threading.Event | None = None, on_progress=None, on_log=None) -> Handle:
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
        ts = _utc_now_iso()
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
            make_payload=lambda buf, chunk, t: {"t_sec": t, "n_samples": int(buf.size)},
        )
        rx = runner.run()

        _emit_progress(on_progress, "export", message="writing wav", meta={"chunk": 0, "data": {}})
        _wav_write(out_path, rx, sr)

        # optional run summary csv
        try:
            cp = outdir / "loopback_record.csv"
            _write_csv_with_meta(
                cp,
                {
                    "schema": "audio_toolkit_csv_v1",
                    "feature": "loopback_record",
                    "mode": "loopback_realtime",
                    "run_id": run_id,
                    "timestamp_utc": ts,
                    "sample_rate": sr,
                    "channels": in_ch,
                    "input_wav_path": str(inp),
                    "output_path": str(out_path),
                    "input_device": request.input_device,
                    "output_device": request.output_device,
                    "blocksize": request.blocksize,
                },
                ["field", "value"],
                [("input_wav_path", str(inp)), ("output_path", str(out_path)), ("input_device", request.input_device), ("output_device", request.output_device)],
            )
        except Exception:
            pass

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(
            feature="loopback_record",
            mode="loopback_realtime",
            summary={"output_path": str(out_path), "fs": sr, "channels": in_ch},
            artifacts=[Artifact(kind="wav", path=str(out_path), description="Recorded loopback WAV", meta={"schema":"audio_toolkit_csv_v1"})],
        )

    return _start_async("loopback_record", impl, stop_event)

# ============================================================
# THD loopback (async)
# ============================================================

def start_thd_loopback(request: ThdLoopbackRequest, *, stop_event: threading.Event | None = None, on_progress=None, on_log=None) -> Handle:
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

        stream_rows: list[tuple[int, float, int, str, str]] = []

        def payload(buf: np.ndarray, chunk: int, t_sec: float) -> dict[str, Any]:
            if buf.size < 2048:
                return {"t_sec": t_sec, "n_samples": int(buf.size), "freq_hz": float(request.freq_hz), "hmax": int(request.hmax), "freq_axis_hz": [], "mag_db": []}
            f, db = _spectrum_db(buf, sr)
            step = max(1, len(f) // 512)
            f_ds = f[::step].astype(float).tolist()
            db_ds = db[::step].astype(float).tolist()
            # store for replay CSV
            stream_rows.append((int(chunk), float(t_sec), int(buf.size), json_dumps(f_ds), json_dumps(db_ds)))
            return {"t_sec": t_sec, "n_samples": int(buf.size), "freq_hz": float(request.freq_hz), "hmax": int(request.hmax), "freq_axis_hz": f_ds, "mag_db": db_ds}

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
            arts.append(Artifact(kind="wav", path=str(wp), description="Recorded THD capture", meta={"schema":"audio_toolkit_csv_v1"}))

        if request.export_csv:
            cp = outdir / "thd_harmonics.csv"
            _emit_progress(on_progress, "export", message="export csv", meta={"chunk": 0, "data": {}})
            _write_csv_with_meta(
                cp,
                {"schema":"audio_toolkit_csv_v1","feature":"thd","mode":"loopback_realtime","run_id":run_id,"timestamp_utc":ts,"sample_rate":sr,"channels":ch,"freq_hz":request.freq_hz,"hmax":request.hmax,"thd_ratio":float(thd),"thd_percent":float(100*thd),"thd_db":float(20*math.log10(thd+1e-15))},
                ["harmonic","freq_hz","mag_db"],
                [(i+1,fk,dbv) for i,(fk,dbv) in enumerate(harms)],
            )
            arts.append(Artifact(kind="csv", path=str(cp), description="THD harmonics", meta={"schema":"audio_toolkit_csv_v1","columns":["harmonic","freq_hz","mag_db"]}))

            # stream replay csv
            sp = outdir / "thd_spectrum_stream.csv"
            _write_csv_with_meta(
                sp,
                {"schema":"audio_toolkit_csv_v1","feature":"thd","mode":"loopback_realtime","run_id":run_id,"timestamp_utc":ts,"sample_rate":sr,"channels":ch,"freq_hz":request.freq_hz,"hmax":request.hmax,"blocksize":request.blocksize},
                ["chunk","t_sec","n_samples","f_axis_hz_json","mag_db_json"],
                stream_rows,
            )
            arts.append(Artifact(kind="csv", path=str(sp), description="THD streaming spectrum (replay)", meta={"schema":"audio_toolkit_csv_v1","columns":["chunk","t_sec","n_samples","f_axis_hz_json","mag_db_json"]}))

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(feature="thd", mode="loopback_realtime", summary={"fs":sr,"freq_hz":float(request.freq_hz),"hmax":int(request.hmax),"thd_ratio":float(thd),"thd_percent":float(100*thd),"thd_db":float(20*math.log10(thd+1e-15))}, artifacts=arts)

    return _start_async("thd_loopback", impl, stop_event)

# ============================================================
# Compressor loopback (async)
# ============================================================

def start_compressor_loopback(request: CompressorLoopbackRequest, *, stop_event: threading.Event | None = None, on_progress=None, on_log=None) -> Handle:
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

        stream_rows: list[tuple[int,float,int,float,int,float,float]] = []  # chunk,t_sec,n_samples,rms_db,step_index,target_amp,target_in_db

        def payload(buf: np.ndarray, chunk: int, t_sec: float) -> dict[str, Any]:
            win = max(64, int(0.05 * sr))
            seg = buf[-win:] if buf.size >= win else buf
            r = float(np.sqrt(np.mean(seg * seg) + 1e-12))
            rms_db = 20.0 * math.log10(r + 1e-12)
            step_index = int(min(len(levels)-1, (buf.size // max(1, step_n))))
            target_amp = float(levels[step_index]) if levels else 0.0
            target_in_db = 20.0 * math.log10(abs(target_amp)/math.sqrt(2)+1e-12)
            stream_rows.append((int(chunk), float(t_sec), int(buf.size), float(rms_db), int(step_index), float(target_amp), float(target_in_db)))
            return {"t_sec": t_sec, "n_samples": int(buf.size), "freq_hz": float(request.freq_hz), "step_duration_s": float(request.step_duration_s), "step_index": step_index, "target_amp": target_amp, "target_in_db": target_in_db, "rms_db": rms_db}

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
            in_db.append(rin); out_db.append(rout); gr_db.append(max(0.0, rin - rout))

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
            arts.append(Artifact(kind="wav", path=str(wp), description="Recorded compressor capture", meta={"schema":"audio_toolkit_csv_v1"}))

        if request.export_csv:
            cp = outdir / "compressor_curve.csv"
            _emit_progress(on_progress, "export", message="export csv", meta={"chunk": 0, "data": {}})
            _write_csv_with_meta(
                cp,
                {"schema":"audio_toolkit_csv_v1","feature":"compressor","mode":"loopback_realtime","run_id":run_id,"timestamp_utc":ts,"sample_rate":sr,"channels":ch,"freq_hz":request.freq_hz,"thr_db":thr_db,"ratio":ratio,"gain_offset_db":gain_offset_db,"levels_count":len(levels),"step_duration_s":request.step_duration_s},
                ["step_index","in_level_db","out_level_db","gain_reduction_db"],
                [(i, a, b, c) for i,(a,b,c) in enumerate(zip(in_db,out_db,gr_db))],
            )
            arts.append(Artifact(kind="csv", path=str(cp), description="Compressor curve", meta={"schema":"audio_toolkit_csv_v1","columns":["step_index","in_level_db","out_level_db","gain_reduction_db"]}))

            sp = outdir / "compressor_stream.csv"
            _write_csv_with_meta(
                sp,
                {"schema":"audio_toolkit_csv_v1","feature":"compressor","mode":"loopback_realtime","run_id":run_id,"timestamp_utc":ts,"sample_rate":sr,"channels":ch,"freq_hz":request.freq_hz,"step_duration_s":request.step_duration_s,"blocksize":request.blocksize},
                ["chunk","t_sec","n_samples","rms_db","step_index","target_amp","target_in_db"],
                stream_rows,
            )
            arts.append(Artifact(kind="csv", path=str(sp), description="Compressor streaming (replay)", meta={"schema":"audio_toolkit_csv_v1","columns":["chunk","t_sec","n_samples","rms_db","step_index","target_amp","target_in_db"]}))

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(feature="compressor", mode="loopback_realtime", summary={"fs": sr, "thr_db": thr_db, "ratio": ratio, "gain_offset_db": gain_offset_db, "no_compression": bool(ratio <= 1.05)}, artifacts=arts)

    return _start_async("compressor_loopback", impl, stop_event)

# ============================================================
# Attack/Release loopback (async)
# ============================================================

def start_attack_release_loopback(request: ARLoopbackRequest, *, stop_event: threading.Event | None = None, on_progress=None, on_log=None) -> Handle:
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

        stream_rows: list[tuple[int,float,int,float,str,str]] = []  # chunk,t_sec,n_samples,t_tail_start,time_json,env_json

        def payload(buf: np.ndarray, chunk: int, t_sec: float) -> dict[str, Any]:
            env_db, t_env = _rms_envelope_db(buf, sr, float(request.rms_win_ms))
            tail_n = min(200, len(env_db))
            t_tail = t_env[-tail_n:]
            e_tail = env_db[-tail_n:]
            t0 = float(t_tail[0]) if t_tail else 0.0
            stream_rows.append((int(chunk), float(t_sec), int(buf.size), t0, json_dumps(t_tail), json_dumps(e_tail)))
            return {"t_sec": t_sec, "n_samples": int(buf.size), "freq_hz": float(request.freq_hz), "rms_win_ms": float(request.rms_win_ms), "t_tail_start_sec": t0, "time_s": t_tail, "envelope_db": e_tail}

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
            arts.append(Artifact(kind="wav", path=str(wp), description="Recorded A/R capture", meta={"schema":"audio_toolkit_csv_v1"}))

        if request.export_csv:
            cp = outdir / "ar_envelope.csv"
            _emit_progress(on_progress, "export", message="export csv", meta={"chunk": 0, "data": {}})
            _write_csv_with_meta(
                cp,
                {"schema":"audio_toolkit_csv_v1","feature":"attack_release","mode":"loopback_realtime","run_id":run_id,"timestamp_utc":ts,"sample_rate":sr,"channels":ch,"freq_hz":request.freq_hz,"rms_win_ms":request.rms_win_ms,"attack_ms":atk,"release_ms":rel},
                ["time_s","envelope_db"],
                [(a, b) for a, b in zip(t_env, env_db)],
            )
            arts.append(Artifact(kind="csv", path=str(cp), description="A/R envelope", meta={"schema":"audio_toolkit_csv_v1","columns":["time_s","envelope_db"]}))

            sp = outdir / "ar_stream.csv"
            _write_csv_with_meta(
                sp,
                {"schema":"audio_toolkit_csv_v1","feature":"attack_release","mode":"loopback_realtime","run_id":run_id,"timestamp_utc":ts,"sample_rate":sr,"channels":ch,"freq_hz":request.freq_hz,"rms_win_ms":request.rms_win_ms,"blocksize":request.blocksize},
                ["chunk","t_sec","n_samples","t_tail_start_sec","time_s_json","envelope_db_json"],
                stream_rows,
            )
            arts.append(Artifact(kind="csv", path=str(sp), description="A/R streaming (replay)", meta={"schema":"audio_toolkit_csv_v1","columns":["chunk","t_sec","n_samples","t_tail_start_sec","time_s_json","envelope_db_json"]}))

        _emit_progress(on_progress, "done", percent=100.0, message="done", meta={"chunk": 0, "data": {}})
        return Result(feature="attack_release", mode="loopback_realtime", summary={"fs": sr, "attack_ms": atk, "release_ms": rel, "rms_win_ms": float(request.rms_win_ms)}, artifacts=arts)

    return _start_async("attack_release_loopback", impl, stop_event)

# ============================================================
# json helper (avoid importing json repeatedly inside callback)
# ============================================================
import json
def json_dumps(x):
    return json.dumps(x, ensure_ascii=False)