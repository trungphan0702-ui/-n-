import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np

# NOTE: GUI is immutable in layout. Backend must be called ONLY via backend.contracts.
from backend.contracts import (
    BackendError, CancelledError,
    # data / events
    ProgressEvent, LogEvent,
    # audio i/o
    ListDevicesRequest, run_list_devices,
    ReadWavRequest, run_wav_read,
    # offline analysis
    ThdOfflineRequest, run_thd_offline,
    CompressorOfflineRequest, run_compressor_offline,
    AROfflineRequest, run_attack_release_offline,
    CompareRequest, run_compare,
    # loopback realtime (async)
    ThdLoopbackRequest, start_thd_loopback,
    CompressorLoopbackRequest, start_compressor_loopback,
    ARLoopbackRequest, start_attack_release_loopback,
    LoopbackRecordRequest, start_loopback_record,
)

# ============================================================
# GUI constants
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCENT = "#0b5cad"
BTN_FONT = ("Segoe UI", 10, "bold")
LOG_FONT = ("Consolas", 10)


# ============================================================
# GUI-local helpers (GUI may use tkinter; backend must not)
# ============================================================
class UILogger:
    def __init__(self, log_fn):
        self._log = log_fn

    def banner(self, msg: str):
        self._log("=" * 60)
        self._log(str(msg))
        self._log("=" * 60)


class PlotWindowManager:
    """
    Placeholder plot manager. Real plot rendering should consume PlotSpec from backend.
    Keeping stubs avoids changing GUI structure later.
    """
    def __init__(self, master, log=None):
        self.master = master
        self.log = log or (lambda *_: None)

    def open_thd_snapshot(self, *args, **kwargs):
        self.log("[plot] THD snapshot (PlotSpec rendering not wired yet)")

    def open_compressor_snapshot(self, *args, **kwargs):
        self.log("[plot] Compressor snapshot (PlotSpec rendering not wired yet)")

    def open_ar_snapshot(self, *args, **kwargs):
        self.log("[plot] A/R snapshot (PlotSpec rendering not wired yet)")

    def open_compare_overlay(self, *args, **kwargs):
        self.log("[plot] Compare overlay (PlotSpec rendering not wired yet)")


class ScrollableFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        canvas = tk.Canvas(self, borderwidth=0, background="#fafafa")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas


# ============================================================
# MAIN APP
# ============================================================
class AudioAnalysisToolkitApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Analysis Suite v3.4 – UI Upgraded")
        master.geometry("1400x900")

        # Variables (keep names consistent with existing GUI wiring)
        self.hw_freq = tk.StringVar(value="1000")
        self.hw_amp = tk.StringVar(value="0.7")
        self.hw_input_dev = tk.StringVar()
        self.hw_output_dev = tk.StringVar()
        self.hw_loop_file = tk.StringVar(value="")
        self.hw_ar_rms_win = tk.StringVar(value="5")
        self.hw_thd_hmax = tk.StringVar(value="5")
        self.thd_max_h = tk.StringVar(value="5")
        self.offline_in = tk.StringVar(value="")
        self.offline_out = tk.StringVar(value="")

        # State: D vs E separation
        self.state = {
            "input_file": "",       # chosen input wav for loopback
            "received_file": "",    # recorded output wav after loopback
        }

        self.stop_event = threading.Event()
        self.worker = None
        self._last_input_devices = []
        self._last_output_devices = []
        self._auto_refresh_job = None
        self.auto_refresh_interval_ms = 8000
        self.auto_refresh_enabled = tk.BooleanVar(value=False)

        self.logger = UILogger(self.hw_log)
        self.plot_manager = PlotWindowManager(master, log=self.hw_log)

        self._configure_style()
        self._build_ui()
        self._refresh_hw_devices()

    # ---------------------------------------------------------
    # STYLE
    # ---------------------------------------------------------
    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background="#fafafa")
        style.configure("TLabelframe", background="#fafafa")
        style.configure("TLabelframe.Label", background="#fafafa", foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("TLabel", background="#fafafa", font=("Segoe UI", 10))
        style.configure("TEntry", padding=4)
        style.configure("Accent.TButton", foreground="white", background=ACCENT, font=BTN_FONT)
        style.map("Accent.TButton", background=[("active", "#094f99")])

    # ---------------------------------------------------------
    # LOG
    # ---------------------------------------------------------
    def hw_log(self, msg: str):
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)

    # ---------------------------------------------------------
    # THREADING (GUI-only)
    # ---------------------------------------------------------
    def _start_thread(self, target, name=None):
        """Run a task in a background thread.
        NOTE: UI updates must be scheduled via master.after(0, ...).
        """
        if self.worker and self.worker.is_alive():
            self.hw_log("Một tác vụ khác đang chạy. Vui lòng chờ...")
            return

        def wrapped():
            try:
                target()
            except Exception as exc:  # best-effort logging
                import traceback

                def dump():
                    self.hw_log(f"[{name or getattr(target, '__name__', 'worker')}] lỗi: {exc}")
                    for line in traceback.format_exc().strip().splitlines():
                        self.hw_log(line)

                try:
                    self.master.after(0, dump)
                except Exception:
                    dump()

        self.worker = threading.Thread(target=wrapped, daemon=True, name=name or "worker")
        self.worker.start()

    def request_stop(self):
        self.stop_event.set()

    # ---------------------------------------------------------
    # INPUT PARSERS
    # ---------------------------------------------------------
    def _parse_float(self, var, default=0.0):
        try:
            return float(var.get())
        except Exception:
            return default

    def _parse_int(self, var, default=1):
        try:
            return int(var.get())
        except Exception:
            return default

    # ---------------------------------------------------------
    # DEVICES
    # ---------------------------------------------------------
    def _require_devices(self) -> bool:
        """Backend-level device check."""
        try:
            _ = run_list_devices(ListDevicesRequest())
            return True
        except Exception as exc:
            messagebox.showerror("Thiết bị âm thanh", f"Không thể lấy danh sách thiết bị.\n{exc}")
            return False

    def _refresh_hw_devices(self, from_timer: bool = False):
        # Ensure UI thread
        if self.master and threading.current_thread() is not threading.main_thread():
            self.master.after(0, lambda: self._refresh_hw_devices(from_timer=from_timer))
            return

        if not self._require_devices():
            return

        try:
            res = run_list_devices(ListDevicesRequest())
            inputs = list(res.summary.get("inputs", []))
            outputs = list(res.summary.get("outputs", []))

            prev_in_sel = self.hw_input_dev.get()
            prev_out_sel = self.hw_output_dev.get()

            added_in = [d for d in inputs if d not in self._last_input_devices]
            removed_in = [d for d in self._last_input_devices if d not in inputs]
            added_out = [d for d in outputs if d not in self._last_output_devices]
            removed_out = [d for d in self._last_output_devices if d not in outputs]

            first_refresh = (not self._last_input_devices) and (not self._last_output_devices)
            changed = bool(added_in or removed_in or added_out or removed_out)

            self._last_input_devices = inputs
            self._last_output_devices = outputs

            self.cb_in["values"] = inputs
            self.cb_out["values"] = outputs

            if changed and not from_timer:
                if added_in:
                    self.hw_log(f"Added inputs: {', '.join(added_in)}")
                if removed_in:
                    self.hw_log(f"Removed inputs: {', '.join(removed_in)}")
                if added_out:
                    self.hw_log(f"Added outputs: {', '.join(added_out)}")
                if removed_out:
                    self.hw_log(f"Removed outputs: {', '.join(removed_out)}")

            # Restore selection if possible
            if prev_in_sel in inputs:
                self.hw_input_dev.set(prev_in_sel)
                self.cb_in.set(prev_in_sel)
            elif inputs:
                self.cb_in.current(0)
                self.hw_input_dev.set(inputs[0])

            if prev_out_sel in outputs:
                self.hw_output_dev.set(prev_out_sel)
                self.cb_out.set(prev_out_sel)
            elif outputs:
                self.cb_out.current(0)
                self.hw_output_dev.set(outputs[0])

            if (changed or first_refresh) and not from_timer:
                self.hw_log("Đã làm mới danh sách thiết bị âm thanh.")
        except Exception as e:
            self.hw_log(f"Lỗi khi lấy thiết bị: {e}")

    def _auto_refresh_tick(self):
        if not self.auto_refresh_enabled.get():
            return
        self._refresh_hw_devices(from_timer=True)
        self._auto_refresh_job = self.master.after(self.auto_refresh_interval_ms, self._auto_refresh_tick)

    def _on_auto_refresh_toggle(self):
        if self._auto_refresh_job:
            self.master.after_cancel(self._auto_refresh_job)
            self._auto_refresh_job = None
        if self.auto_refresh_enabled.get():
            self._auto_refresh_job = self.master.after(self.auto_refresh_interval_ms, self._auto_refresh_tick)

    def _device_indices(self):
        """Parse device index from combobox label. Expected format: '<index>: <name>' or '(none)'."""
        def _idx(s: str):
            s = (s or "").strip()
            if not s or s == "(none)":
                return None
            for sep in (":", "-", "—"):
                if sep in s:
                    left = s.split(sep, 1)[0].strip()
                    try:
                        return int(left)
                    except Exception:
                        break
            try:
                return int(s)
            except Exception:
                return None

        return _idx(self.hw_input_dev.get()), _idx(self.hw_output_dev.get())

    # ---------------------------------------------------------
    # FILE PICKERS
    # ---------------------------------------------------------
    def select_hw_loop_file(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if p:
            self.hw_loop_file.set(p)
            self.state["input_file"] = p
            self.hw_log(f"Đã chọn file: {p}")

    def select_offline_in(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if p:
            self.offline_in.set(p)
            self.hw_log(f"File Input: {p}")

    def select_offline_out(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if p:
            self.offline_out.set(p)
            self.hw_log(f"File Output: {p}")

    # ---------------------------------------------------------
    # BUILD UI (layout kept consistent with your current version)
    # ---------------------------------------------------------
    def _build_ui(self):
        nb = ttk.Notebook(self.master)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab_hw = ttk.Frame(nb)
        nb.add(tab_hw, text="4. Hardware Loopback (Real-time)")

        # Top device frame
        dev_frame = ttk.LabelFrame(tab_hw, text="Cấu hình Soundcard (Input / Output)")
        dev_frame.pack(fill="x", padx=8, pady=6)

        ttk.Label(dev_frame, text="Input Device:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.cb_in = ttk.Combobox(dev_frame, textvariable=self.hw_input_dev, width=60, state="readonly")
        self.cb_in.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(dev_frame, text="Output Device:").grid(row=0, column=2, sticky="e", padx=4)
        self.cb_out = ttk.Combobox(dev_frame, textvariable=self.hw_output_dev, width=60, state="readonly")
        self.cb_out.grid(row=0, column=3, sticky="w", padx=4)

        ttk.Button(dev_frame, text="Làm mới", command=self._refresh_hw_devices).grid(row=0, column=4, padx=6)

        ttk.Checkbutton(
            dev_frame,
            text="Auto refresh devices",
            variable=self.auto_refresh_enabled,
            command=self._on_auto_refresh_toggle
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=4, pady=(2, 0))

        # PanedWindow
        paned = ttk.PanedWindow(tab_hw, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # LEFT (scrollable)
        left_container = ttk.Frame(paned)
        paned.add(left_container, weight=1)

        scroll_left = ScrollableFrame(left_container)
        scroll_left.pack(fill="both", expand=True)
        left = scroll_left.scrollable_frame

        # -------------------------------------------------
        # SECTION A
        grp_a = ttk.LabelFrame(left, text="A. Đo Compressor (Stepped Sweep)")
        grp_a.pack(fill="x", padx=6, pady=8)

        ttk.Label(grp_a, text="Quét 36 mức (0.25s/mức) – Tìm Thr, Ratio, Makeup Gain", foreground="blue").pack(anchor="w", padx=6, pady=4)

        f = ttk.Frame(grp_a)
        f.pack(fill="x", padx=6, pady=4)
        ttk.Label(f, text="Freq (Hz):").pack(side="left")
        ttk.Entry(f, textvariable=self.hw_freq, width=8).pack(side="left", padx=6)

        ttk.Button(
            grp_a,
            text="▶ CHẠY TEST COMPRESSOR (HW)",
            style="Accent.TButton",
            command=lambda: self._start_thread(self.run_hw_compressor, name="compressor_hw"),
        ).pack(fill="x", padx=6, pady=8)

        # -------------------------------------------------
        # SECTION B
        grp_b = ttk.LabelFrame(left, text="B. Đo THD (Harmonic Distortion)")
        grp_b.pack(fill="x", padx=6, pady=8)

        fb = ttk.Frame(grp_b)
        fb.pack(fill="x", padx=6, pady=4)
        ttk.Label(fb, text="Amp (0-1):").pack(side="left")
        ttk.Entry(fb, textvariable=self.hw_amp, width=8).pack(side="left", padx=6)
        ttk.Label(fb, text="Max H:").pack(side="left", padx=(10, 2))
        ttk.Entry(fb, textvariable=self.thd_max_h, width=4).pack(side="left")

        ttk.Button(
            grp_b,
            text="▶ CHẠY TEST THD (HW)",
            command=lambda: self._start_thread(self.run_hw_thd, name="thd_hw"),
        ).pack(fill="x", padx=6, pady=8)

        # -------------------------------------------------
        # SECTION C
        grp_c = ttk.LabelFrame(left, text="C. Đo Attack / Release (Step Tone)")
        grp_c.pack(fill="x", padx=6, pady=8)

        ttk.Button(
            grp_c,
            text="▶ CHẠY TEST A/R (HW)",
            command=lambda: self._start_thread(self.run_hw_attack_release, name="ar_hw"),
        ).pack(fill="x", padx=6, pady=8)

        far = ttk.Frame(grp_c)
        far.pack(fill="x", padx=6, pady=4)
        ttk.Label(far, text="RMS Win (ms):").pack(side="left")
        ttk.Entry(far, textvariable=self.hw_ar_rms_win, width=6).pack(side="left", padx=6)

        # -------------------------------------------------
        # SECTION D  (RESTORED: loopback + analyze received file)
        grp_d = ttk.LabelFrame(left, text="D. Loopback & Phân tích File")
        grp_d.pack(fill="x", padx=6, pady=8)

        ffile = ttk.Frame(grp_d)
        ffile.pack(fill="x", padx=6, pady=6)
        ttk.Label(ffile, text="File WAV Input:").grid(row=0, column=0, sticky="w")
        ttk.Entry(ffile, textvariable=self.hw_loop_file, width=40).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(ffile, text="Browse...", command=self.select_hw_loop_file).grid(row=0, column=2, padx=6)
        ffile.grid_columnconfigure(1, weight=1)

        ttk.Button(
            grp_d,
            text="1. ▶ CHẠY LOOPBACK & SAVE (All Files)",
            style="Accent.TButton",
            command=lambda: self._start_thread(self.run_loopback_record, name="loopback"),
        ).pack(fill="x", padx=6, pady=8)

        # Sub-analysis: analyze the recorded file (received.wav) ONLY
        ana = ttk.LabelFrame(grp_d, text="Phân tích File Ghi âm")
        ana.pack(fill="x", padx=6, pady=4)

        ttk.Button(
            ana,
            text="A. Phân tích Compressor",
            command=lambda: self._start_thread(lambda: self.analyze_loopback("compressor"), name="ana_comp"),
        ).pack(fill="x", padx=6, pady=4)

        f_thd = ttk.Frame(ana)
        f_thd.pack(fill="x", padx=6, pady=4)
        ttk.Button(
            f_thd,
            text="B. Phân tích THD",
            command=lambda: self._start_thread(lambda: self.analyze_loopback("thd"), name="ana_thd"),
        ).pack(side="left", expand=True, fill="x")
        ttk.Label(f_thd, text="Max H:").pack(side="left", padx=6)
        ttk.Entry(f_thd, textvariable=self.hw_thd_hmax, width=4).pack(side="left")

        f_ar2 = ttk.Frame(ana)
        f_ar2.pack(fill="x", padx=6, pady=4)
        ttk.Button(
            f_ar2,
            text="C. Phân tích A/R",
            command=lambda: self._start_thread(lambda: self.analyze_loopback("ar"), name="ana_ar"),
        ).pack(side="left", expand=True, fill="x")
        ttk.Label(f_ar2, text="RMS Win (ms):").pack(side="left", padx=6)
        ttk.Entry(f_ar2, textvariable=self.hw_ar_rms_win, width=4).pack(side="left")

        # -------------------------------------------------
        # SECTION E
        grp_e = ttk.LabelFrame(left, text="E. Phân tích 2 File Offline")
        grp_e.pack(fill="x", padx=6, pady=8)

        fe = ttk.Frame(grp_e)
        fe.pack(fill="x", padx=6, pady=6)

        ttk.Label(fe, text="File Input:").grid(row=0, column=0)
        ttk.Entry(fe, width=30, textvariable=self.offline_in).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(fe, text="Browse...", command=self.select_offline_in).grid(row=0, column=2, padx=6)

        ttk.Label(fe, text="File Output:").grid(row=1, column=0)
        ttk.Entry(fe, width=30, textvariable=self.offline_out).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(fe, text="Browse...", command=self.select_offline_out).grid(row=1, column=2, padx=6)
        fe.grid_columnconfigure(1, weight=1)

        small_ana = ttk.Frame(grp_e)
        small_ana.pack(fill="x", padx=6, pady=4)
        ttk.Button(
            small_ana,
            text="A. Phân tích Compressor",
            command=lambda: self._start_thread(lambda: self.analyze_offline("compressor"), name="off_comp"),
        ).pack(fill="x", pady=2)
        ttk.Button(
            small_ana,
            text="B. Phân tích THD",
            command=lambda: self._start_thread(lambda: self.analyze_offline("thd"), name="off_thd"),
        ).pack(fill="x", pady=2)
        ttk.Button(
            small_ana,
            text="C. Phân tích A/R",
            command=lambda: self._start_thread(lambda: self.analyze_offline("ar"), name="off_ar"),
        ).pack(fill="x", pady=2)

        # -------------------------------------------------
        # RIGHT PANEL: LOGS
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        ttk.Label(right, text="Nhật ký (Logs):", background="#fafafa").pack(anchor="w", padx=6)

        log_frame = ttk.Frame(right)
        log_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.log_text = tk.Text(log_frame, font=LOG_FONT, bg="#f4f4f4", wrap="none")
        self.log_text.pack(side="left", fill="both", expand=True)

        scroll_log = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll_log.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll_log.set)

        # Startup log
        self.hw_log("[Khởi động] Làm mới danh sách thiết bị âm thanh.")

    # ---------------------------------------------------------
    # LOOPBACK REALTIME (A/B/C)
    # ---------------------------------------------------------
    def run_hw_thd(self):
        """HW THD loopback via backend contract (async)."""
        if not self._require_devices():
            return

        freq = self._parse_float(self.hw_freq, 1000.0)
        amp = self._parse_float(self.hw_amp, 0.7)
        hmax = self._parse_int(self.thd_max_h, 5)
        in_dev, out_dev = self._device_indices()

        req = ThdLoopbackRequest(
            freq_hz=freq,
            amp=amp,
            hmax=hmax,
            input_device=in_dev,
            output_device=out_dev,
        )

        self.logger.banner(f"THD LOOPBACK @ {freq}Hz | amp={amp} | hmax={hmax}")
        self.stop_event.clear()

        def on_log(ev: LogEvent):
            self.master.after(0, lambda: self.hw_log(ev.message))

        def on_progress(ev: ProgressEvent):
            # Required: meta['chunk'] for streaming
            if ev.phase == "streaming":
                chunk = (ev.meta or {}).get("chunk", None)
                msg = f"[streaming] chunk={chunk}" if chunk is not None else "[streaming]"
            else:
                msg = f"[{ev.phase}] {ev.message}"
            self.master.after(0, lambda: self.hw_log(msg))

        def work():
            try:
                handle = start_thd_loopback(req, stop_event=self.stop_event, on_progress=on_progress, on_log=on_log)
                result = handle.join()
            except CancelledError:
                self.master.after(0, lambda: self.hw_log("Đã dừng (cancel)."))
                return
            except BackendError as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi THD: {exc}"))
                return
            except Exception as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi không xác định: {exc}"))
                return

            def show():
                s = result.summary or {}
                thd_p = s.get("thd_percent")
                thd_db = s.get("thd_db")
                if thd_p is not None:
                    try:
                        self.hw_log(f"THD: {float(thd_p):.4f}% | {float(thd_db or 0.0):.2f} dB")
                    except Exception:
                        self.hw_log(f"THD: {thd_p} | {thd_db}")
                for art in getattr(result, "artifacts", []):
                    self.hw_log(f"Artifact: {art.kind} -> {art.path}")

            self.master.after(0, show)

        self._start_thread(work, name="thd_loopback")

    def run_hw_compressor(self):
        """HW Compressor loopback via backend contract (async)."""
        if not self._require_devices():
            return

        freq = self._parse_float(self.hw_freq, 1000.0)
        amp_max = self._parse_float(self.hw_amp, 1.36)
        in_dev, out_dev = self._device_indices()

        req = CompressorLoopbackRequest(
            freq_hz=freq,
            amp_max=amp_max,
            input_device=in_dev,
            output_device=out_dev,
        )

        self.logger.banner(f"COMPRESSOR LOOPBACK @ {freq}Hz | amp_max={amp_max}")
        self.stop_event.clear()

        def on_log(ev: LogEvent):
            self.master.after(0, lambda: self.hw_log(ev.message))

        def on_progress(ev: ProgressEvent):
            if ev.phase == "streaming":
                chunk = (ev.meta or {}).get("chunk", None)
                msg = f"[streaming] chunk={chunk}" if chunk is not None else "[streaming]"
            else:
                msg = f"[{ev.phase}] {ev.message}"
            self.master.after(0, lambda: self.hw_log(msg))

        def work():
            try:
                handle = start_compressor_loopback(req, stop_event=self.stop_event, on_progress=on_progress, on_log=on_log)
                result = handle.join()
            except CancelledError:
                self.master.after(0, lambda: self.hw_log("Đã dừng (cancel)."))
                return
            except BackendError as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi compressor: {exc}"))
                return
            except Exception as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi không xác định: {exc}"))
                return

            def show():
                s = result.summary or {}
                if s.get("no_compression"):
                    self.hw_log("No compression.")
                else:
                    thr = s.get("thr_db")
                    ratio = s.get("ratio")
                    go = s.get("gain_offset_db")
                    if thr is not None and ratio is not None:
                        self.hw_log(f"Thr {thr:.2f} dB | Ratio {ratio:.2f}:1 | Gain {go:+.2f} dB")
                for art in getattr(result, "artifacts", []):
                    self.hw_log(f"Artifact: {art.kind} -> {art.path}")

            self.master.after(0, show)

        self._start_thread(work, name="compressor_loopback")

    def run_hw_attack_release(self):
        """HW Attack/Release loopback via backend contract (async)."""
        if not self._require_devices():
            return

        freq = self._parse_float(self.hw_freq, 1000.0)
        amp = self._parse_float(self.hw_amp, 0.7)
        rms_win = self._parse_float(self.hw_ar_rms_win, 5.0)
        in_dev, out_dev = self._device_indices()

        req = ARLoopbackRequest(
            freq_hz=freq,
            amp=amp,
            rms_win_ms=rms_win,
            input_device=in_dev,
            output_device=out_dev,
        )

        self.logger.banner(f"A/R LOOPBACK @ {freq}Hz | rms_win={rms_win}ms")
        self.stop_event.clear()

        def on_log(ev: LogEvent):
            self.master.after(0, lambda: self.hw_log(ev.message))

        def on_progress(ev: ProgressEvent):
            if ev.phase == "streaming":
                chunk = (ev.meta or {}).get("chunk", None)
                msg = f"[streaming] chunk={chunk}" if chunk is not None else "[streaming]"
            else:
                msg = f"[{ev.phase}] {ev.message}"
            self.master.after(0, lambda: self.hw_log(msg))

        def work():
            try:
                handle = start_attack_release_loopback(req, stop_event=self.stop_event, on_progress=on_progress, on_log=on_log)
                result = handle.join()
            except CancelledError:
                self.master.after(0, lambda: self.hw_log("Đã dừng (cancel)."))
                return
            except BackendError as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi A/R: {exc}"))
                return
            except Exception as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi không xác định: {exc}"))
                return

            def show():
                s = result.summary or {}
                atk = s.get("attack_ms")
                rel = s.get("release_ms")
                if atk is not None and rel is not None:
                    self.hw_log(f"Attack {atk:.1f} ms | Release {rel:.1f} ms")
                for art in getattr(result, "artifacts", []):
                    self.hw_log(f"Artifact: {art.kind} -> {art.path}")

            self.master.after(0, show)

        self._start_thread(work, name="ar_loopback")

    # ---------------------------------------------------------
    # D1: LOOPBACK RECORD (SAVE)
    # ---------------------------------------------------------
    def run_loopback_record(self):
        """Play input WAV through output device and record to received.wav (async)."""
        if not self._require_devices():
            return

        in_path = self.hw_loop_file.get().strip()
        if not in_path or not os.path.isfile(in_path):
            messagebox.showwarning("Loopback", "Chọn file WAV hợp lệ trước khi chạy loopback.")
            return

        in_dev, out_dev = self._device_indices()
        out_path = os.path.join(BASE_DIR, "received.wav")

        req = LoopbackRecordRequest(
            input_wav_path=in_path,
            input_device=in_dev,
            output_device=out_dev,
            output_path=out_path,
            input_channels=1,
        )

        self.logger.banner("Chạy loopback & save received.wav")
        self.stop_event.clear()

        def on_log(ev: LogEvent):
            self.master.after(0, lambda: self.hw_log(ev.message))

        def on_progress(ev: ProgressEvent):
            self.master.after(0, lambda: self.hw_log(f"[{ev.phase}] {ev.message}"))

        def work():
            try:
                handle = start_loopback_record(req, stop_event=self.stop_event, on_progress=on_progress, on_log=on_log)
                result = handle.join()
            except CancelledError:
                self.master.after(0, lambda: self.hw_log("Đã dừng (cancel)."))
                return
            except BackendError as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi loopback: {exc}"))
                return
            except Exception as exc:
                self.master.after(0, lambda: self.hw_log(f"Lỗi không xác định: {exc}"))
                return

            def show():
                # D source of truth
                self.state["received_file"] = out_path
                self.hw_log(f"[D] Đã ghi xong: {out_path}")
                for art in getattr(result, "artifacts", []):
                    self.hw_log(f"Artifact: {art.kind} -> {art.path}")

                # OPTIONAL UX: auto-fill E output with received.wav (does NOT change E rule)
                try:
                    self.offline_out.set(out_path)
                    self.hw_log("[UX] Đã tự điền File Output (E) = received.wav (có thể đổi lại nếu muốn).")
                except Exception:
                    pass

            self.master.after(0, show)

        self._start_thread(work, name="loopback_record")

    # ---------------------------------------------------------
    # D2: ANALYZE RECORDED FILE (received.wav ONLY)
    # ---------------------------------------------------------
    def analyze_loopback(self, kind: str):
        """Analyze the recorded file from section D ONLY."""
        rx = (self.state.get("received_file") or "").strip()
        if not rx or not os.path.isfile(rx):
            messagebox.showwarning("Phân tích (D)", "Chưa có file ghi âm loopback (received.wav). Hãy chạy LOOPBACK & SAVE trước.")
            return

        self.hw_log(f"[D] Analyze '{kind}' from received: {rx}")

        try:
            if kind == "compressor":
                res = run_compressor_offline(CompressorOfflineRequest(wav_path=rx, freq_hz=self._parse_float(self.hw_freq, 1000.0)))
                s = res.summary or {}
                if s.get("no_compression"):
                    self.hw_log("[D] Compressor: No compression.")
                else:
                    self.hw_log(f"[D] Compressor: Thr {s.get('thr_db')} dB | Ratio {s.get('ratio')} | Gain {s.get('gain_offset_db')} dB")
                for art in getattr(res, "artifacts", []):
                    self.hw_log(f"[D] Artifact: {art.kind} -> {art.path}")

            elif kind == "thd":
                hmax = self._parse_int(self.hw_thd_hmax, 5)
                res = run_thd_offline(ThdOfflineRequest(wav_path=rx, freq_hz=self._parse_float(self.hw_freq, 1000.0), hmax=hmax))
                s = res.summary or {}
                self.hw_log(f"[D] THD: {s.get('thd_percent')}% | {s.get('thd_db')} dB (hmax={hmax})")
                for art in getattr(res, "artifacts", []):
                    self.hw_log(f"[D] Artifact: {art.kind} -> {art.path}")

            elif kind == "ar":
                rms_win = self._parse_float(self.hw_ar_rms_win, 5.0)
                res = run_attack_release_offline(AROfflineRequest(wav_path=rx, rms_win_ms=rms_win))
                s = res.summary or {}
                self.hw_log(f"[D] A/R: attack={s.get('attack_ms')} ms | release={s.get('release_ms')} ms (rms={rms_win}ms)")
                for art in getattr(res, "artifacts", []):
                    self.hw_log(f"[D] Artifact: {art.kind} -> {art.path}")

            else:
                self.hw_log(f"[D] Unknown analysis kind: {kind}")

        except BackendError as exc:
            self.hw_log(f"[D] Lỗi phân tích: {exc}")
        except Exception as exc:
            self.hw_log(f"[D] Lỗi không xác định: {exc}")

    # ---------------------------------------------------------
    # E: OFFLINE ANALYSIS (use offline_in/offline_out ONLY)
    # ---------------------------------------------------------
    def analyze_offline(self, kind: str):
        """Analyze offline based on section E inputs ONLY."""
        in_path = (self.offline_in.get() or "").strip()
        out_path = (self.offline_out.get() or "").strip()

        # E rules:
        # - THD/Compressor/A-R: default analyze Output file (measured).
        # - Compare: needs both (not wired by UI buttons right now).
        target = out_path if out_path else in_path

        if not target or not os.path.isfile(target):
            messagebox.showwarning("Phân tích (E)", "Chọn file Offline hợp lệ trước (Input/Output).")
            return

        self.hw_log(f"[E] Analyze '{kind}' using offline paths: IN={in_path or '(empty)'} | OUT={out_path or '(empty)'}")
        self.hw_log(f"[E] Target file: {target}")

        try:
            if kind == "compressor":
                res = run_compressor_offline(CompressorOfflineRequest(wav_path=target, freq_hz=self._parse_float(self.hw_freq, 1000.0)))
                s = res.summary or {}
                if s.get("no_compression"):
                    self.hw_log("[E] Compressor: No compression.")
                else:
                    self.hw_log(f"[E] Compressor: Thr {s.get('thr_db')} dB | Ratio {s.get('ratio')} | Gain {s.get('gain_offset_db')} dB")
                for art in getattr(res, "artifacts", []):
                    self.hw_log(f"[E] Artifact: {art.kind} -> {art.path}")

            elif kind == "thd":
                hmax = self._parse_int(self.thd_max_h, 5)
                res = run_thd_offline(ThdOfflineRequest(wav_path=target, freq_hz=self._parse_float(self.hw_freq, 1000.0), hmax=hmax))
                s = res.summary or {}
                self.hw_log(f"[E] THD: {s.get('thd_percent')}% | {s.get('thd_db')} dB (hmax={hmax})")
                for art in getattr(res, "artifacts", []):
                    self.hw_log(f"[E] Artifact: {art.kind} -> {art.path}")

            elif kind == "ar":
                rms_win = self._parse_float(self.hw_ar_rms_win, 5.0)
                res = run_attack_release_offline(AROfflineRequest(wav_path=target, rms_win_ms=rms_win))
                s = res.summary or {}
                self.hw_log(f"[E] A/R: attack={s.get('attack_ms')} ms | release={s.get('release_ms')} ms (rms={rms_win}ms)")
                for art in getattr(res, "artifacts", []):
                    self.hw_log(f"[E] Artifact: {art.kind} -> {art.path}")

            else:
                self.hw_log(f"[E] Unknown analysis kind: {kind}")

        except BackendError as exc:
            self.hw_log(f"[E] Lỗi phân tích: {exc}")
        except Exception as exc:
            self.hw_log(f"[E] Lỗi không xác định: {exc}")


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalysisToolkitApp(root)
    root.mainloop()
