import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np

# Try to import sounddevice for device listing
try:
    import sounddevice as sd
except Exception:
    sd = None

from analysis import attack_release, compare, compressor, thd, live_measurements
from audio import devices, playrec, wav_io
from utils.logging import UILogger
from utils.plot_windows import PlotWindowManager
from utils.threading import run_in_thread

# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
ACCENT = '#0b66c3'
BTN_FONT = (None, 10, 'bold')
LOG_FONT = ('Consolas', 10)

# ============================================================
# Scrollable Frame Class (LEFT PANEL)
# ============================================================
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

        # Variables
        self.hw_freq = tk.StringVar(value='1000')
        self.hw_amp = tk.StringVar(value='0.7')
        self.hw_input_dev = tk.StringVar()
        self.hw_output_dev = tk.StringVar()
        self.hw_loop_file = tk.StringVar(value='')
        self.hw_ar_rms_win = tk.StringVar(value='5')
        self.hw_thd_hmax = tk.StringVar(value='5')
        self.thd_max_h = tk.StringVar(value='5')
        self.offline_in = tk.StringVar(value='')
        self.offline_out = tk.StringVar(value='')

        self.state = {'input_file': '', 'received_file': ''}
        self.stop_event = threading.Event()
        self.worker = None
        self._last_input_devices = []
        self._last_output_devices = []
        self._devices_signature = None
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
        style.configure("TLabelframe.Label", background="#fafafa", foreground=ACCENT, font=('Segoe UI', 11, 'bold'))
        style.configure("TLabel", background="#fafafa", font=('Segoe UI', 10))
        style.configure("TEntry", padding=4)
        style.configure("Accent.TButton", foreground="white", background=ACCENT, font=BTN_FONT)
        style.map("Accent.TButton", background=[("active", "#094f99")])

    # ---------------------------------------------------------
    # LOG
    # ---------------------------------------------------------
    def hw_log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    # ---------------------------------------------------------
    def _start_thread(self, target, name=None):
        if self.worker and self.worker.is_alive():
            self.hw_log("Một tác vụ khác đang chạy. Vui lòng chờ...")
            return
        def wrapped():
            try:
                target()
            except Exception as exc:
                import traceback

                self.hw_log(f"[{name or target.__name__}] lỗi: {exc}")
                for line in traceback.format_exc().strip().splitlines():
                    self.hw_log(line)

        self.worker = run_in_thread(wrapped, self.stop_event, name=name)

    def request_stop(self):
        self.stop_event.set()

    def _schedule_plot(self, func, *args, **kwargs):
        if self.master:
            self.master.after(0, lambda: func(*args, **kwargs))

    def _require_sounddevice(self):
        if sd is None:
            messagebox.showerror("Sounddevice", "Sounddevice không khả dụng. Cài đặt thư viện trước khi đo HW.")
            return False
        return True

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
    # DEVICE REFRESH
    # ---------------------------------------------------------
    def _refresh_hw_devices(self, from_timer: bool = False):
        if self.master and threading.current_thread() is not threading.main_thread():
            self.master.after(0, lambda: self._refresh_hw_devices(from_timer=from_timer))
            return

        if sd is None:
            self.hw_log("Sounddevice không khả dụng.")
            return
        try:
            raw_devices = sd.query_devices()
            inputs, outputs = devices.list_devices(raise_on_error=True)

            in_count = sum(1 for d in raw_devices if d.get('max_input_channels', 0) > 0)
            out_count = sum(1 for d in raw_devices if d.get('max_output_channels', 0) > 0)

            prev_in_sel = self.hw_input_dev.get()
            prev_out_sel = self.hw_output_dev.get()

            added_in = [d for d in inputs if d not in self._last_input_devices]
            removed_in = [d for d in self._last_input_devices if d not in inputs]
            added_out = [d for d in outputs if d not in self._last_output_devices]
            removed_out = [d for d in self._last_output_devices if d not in outputs]

            changed = bool(added_in or removed_in or added_out or removed_out)
            first_refresh = not self._devices_signature

            if changed or first_refresh:
                self.cb_in['values'] = inputs
                self.cb_out['values'] = outputs

                if added_in:
                    self.hw_log(f"Added inputs: {', '.join(added_in)}")
                if removed_in:
                    self.hw_log(f"Removed inputs: {', '.join(removed_in)}")
                if added_out:
                    self.hw_log(f"Added outputs: {', '.join(added_out)}")
                if removed_out:
                    self.hw_log(f"Removed outputs: {', '.join(removed_out)}")

                if prev_in_sel in inputs:
                    self.hw_input_dev.set(prev_in_sel)
                    self.cb_in.set(prev_in_sel)
                elif inputs:
                    self.cb_in.current(0)
                    self.hw_input_dev.set(inputs[0])
                    if prev_in_sel:
                        self.hw_log(f"Input '{prev_in_sel}' không còn khả dụng, chuyển sang {inputs[0]}")

                if prev_out_sel in outputs:
                    self.hw_output_dev.set(prev_out_sel)
                    self.cb_out.set(prev_out_sel)
                elif outputs:
                    self.cb_out.current(0)
                    self.hw_output_dev.set(outputs[0])
                    if prev_out_sel:
                        self.hw_log(f"Output '{prev_out_sel}' không còn khả dụng, chuyển sang {outputs[0]}")

            selected_in = self.hw_input_dev.get() or "(none)"
            selected_out = self.hw_output_dev.get() or "(none)"

            if changed or first_refresh:
                self.hw_log(
                    "Đã làm mới danh sách thiết bị âm thanh. "
                    f"Inputs: {len(inputs)}, Outputs: {len(outputs)} (query_devices: {in_count}/{out_count}). "
                    f"Chọn input: {selected_in}; chọn output: {selected_out}."
                )
            elif not from_timer:
                self.hw_log(
                    "Danh sách thiết bị không đổi. "
                    f"Inputs: {len(inputs)}, Outputs: {len(outputs)} (query_devices: {in_count}/{out_count}). "
                    f"Chọn input: {selected_in}; chọn output: {selected_out}."
                )

            self._last_input_devices = inputs
            self._last_output_devices = outputs
            self._devices_signature = devices.get_devices_signature()
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

    # ---------------------------------------------------------
    def select_hw_loop_file(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if p:
            self.hw_loop_file.set(p)
            self.state['input_file'] = p
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
    # BUILD UI
    # ---------------------------------------------------------
    def _build_ui(self):

        # Notebook
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

        ttk.Button(grp_a, text="▶ CHẠY TEST COMPRESSOR (HW)",
                   style="Accent.TButton",
                   command=lambda: self._start_thread(self.run_hw_compressor, name="compressor_hw")
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

        ttk.Button(grp_b, text="▶ CHẠY TEST THD (HW)",
                   command=lambda: self._start_thread(self.run_hw_thd, name="thd_hw")
                   ).pack(fill="x", padx=6, pady=8)

        # -------------------------------------------------
        # SECTION C
        grp_c = ttk.LabelFrame(left, text="C. Đo Attack / Release (Step Tone)")
        grp_c.pack(fill="x", padx=6, pady=8)

        ttk.Button(grp_c, text="▶ CHẠY TEST A/R (HW)",
                   command=lambda: self._start_thread(self.run_hw_attack_release, name="ar_hw")
                   ).pack(fill="x", padx=6, pady=8)

        far = ttk.Frame(grp_c)
        far.pack(fill="x", padx=6, pady=4)
        ttk.Label(far, text="RMS Win (ms):").pack(side="left")
        ttk.Entry(far, textvariable=self.hw_ar_rms_win, width=6).pack(side="left", padx=6)

        # -------------------------------------------------
        # SECTION D
        grp_d = ttk.LabelFrame(left, text="D. Loopback & Phân tích File")
        grp_d.pack(fill="x", padx=6, pady=8)

        ffile = ttk.Frame(grp_d)
        ffile.pack(fill="x", padx=6, pady=6)
        ttk.Label(ffile, text="File WAV Input:").grid(row=0, column=0, sticky="w")
        ttk.Entry(ffile, textvariable=self.hw_loop_file, width=40).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(ffile, text="Browse...", command=self.select_hw_loop_file).grid(row=0, column=2, padx=6)
        ffile.grid_columnconfigure(1, weight=1)

        ttk.Button(grp_d, text="1. ▶ CHẠY LOOPBACK & SAVE (All Files)",
                   style="Accent.TButton",
                   command=lambda: self._start_thread(self.run_loopback_record, name="loopback")
                   ).pack(fill="x", padx=6, pady=8)

        # Sub-analysis
        ana = ttk.LabelFrame(grp_d, text="Phân tích File Ghi âm")
        ana.pack(fill="x", padx=6, pady=4)

        ttk.Button(ana, text="A. Phân tích Compressor",
                   command=lambda: self._start_thread(lambda: self.analyze_loopback('compressor'), name="ana_comp")).pack(fill="x", padx=6, pady=4)

        f_thd = ttk.Frame(ana)
        f_thd.pack(fill="x", padx=6, pady=4)
        ttk.Button(f_thd, text="B. Phân tích THD",
                   command=lambda: self._start_thread(lambda: self.analyze_loopback('thd'), name="ana_thd")
                   ).pack(side="left", expand=True, fill="x")
        ttk.Label(f_thd, text="Max H:").pack(side="left", padx=6)
        ttk.Entry(f_thd, textvariable=self.hw_thd_hmax, width=4).pack(side="left")

        f_ar2 = ttk.Frame(ana)
        f_ar2.pack(fill="x", padx=6, pady=4)
        ttk.Button(f_ar2, text="C. Phân tích A/R",
                   command=lambda: self._start_thread(lambda: self.analyze_loopback('ar'), name="ana_ar")).pack(side="left", expand=True, fill="x")
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
        ttk.Button(small_ana, text="A. Phân tích Compressor", command=lambda: self._start_thread(lambda: self.analyze_offline('compressor'), name="off_comp")).pack(fill="x", pady=2)
        ttk.Button(small_ana, text="B. Phân tích THD", command=lambda: self._start_thread(lambda: self.analyze_offline('thd'), name="off_thd")).pack(fill="x", pady=2)
        ttk.Button(small_ana, text="C. Phân tích A/R", command=lambda: self._start_thread(lambda: self.analyze_offline('ar'), name="off_ar")).pack(fill="x", pady=2)

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
    # HARDWARE TASKS
    # ---------------------------------------------------------
    def _device_indices(self):
        return devices.parse_device(self.hw_input_dev.get()), devices.parse_device(self.hw_output_dev.get())

    def _prepare_signal(self, sig: np.ndarray) -> np.ndarray:
        sig = sig.astype(np.float32)
        fade = min(256, len(sig) // 10)
        if fade > 0:
            window = np.linspace(0, 1, fade)
            sig[:fade] *= window
            sig[-fade:] *= window[::-1]
        return sig

    def _log_recording_stats(self, data: np.ndarray, label: str = "Rx"):
        try:
            flat = np.asarray(data, dtype=np.float32).flatten()
            if flat.size == 0:
                self.hw_log(f"{label}: (no samples)")
                return
            rms = float(np.sqrt(np.mean(np.square(flat))))
            clip = int(np.sum(np.abs(flat) >= 0.999))
            self.hw_log(f"{label}: min {flat.min():.4f} | max {flat.max():.4f} | rms {rms:.4f} | clips {clip}")
        except Exception as exc:  # pragma: no cover - logging best-effort
            self.hw_log(f"{label}: lỗi khi log thống kê ({exc})")

    def run_hw_thd(self):
        if not self._require_sounddevice():
            return
        freq = self._parse_float(self.hw_freq, 1000.0)
        amp = self._parse_float(self.hw_amp, 0.7)
        hmax = self._parse_int(self.thd_max_h, 5)
        in_dev, out_dev = self._device_indices()
        fs = devices.default_samplerate(out_dev or None)
        tone = self._prepare_signal(live_measurements.generate_thd_tone(freq, amp, fs))
        self.logger.banner(f"THD HW @ {freq} Hz, amp {amp}")
        recorded = playrec.play_and_record(
            tone, fs, in_dev, out_dev, self.stop_event, log=self.hw_log, input_channels=1
        )
        if recorded is None or len(recorded) == 0:
            self.hw_log("Không ghi được dữ liệu.")
            return
        self._log_recording_stats(recorded, "THD Rx")
        res = live_measurements.analyze_thd_capture(recorded, fs, freq, hmax)
        thd_percent = res.get("thd_percent_manual", 0.0)
        thd_db = res.get("thd_db_manual", res.get("thd_db", 0.0))
        harmonics = res.get("harmonics_manual", {})
        self.hw_log(f"THD ≈ {thd_percent:.4f}% ({thd_db:.2f} dB)")
        for h, v in harmonics.items():
            self.hw_log(f"H{h}: {v:.2f} dBc")
        csv_path = live_measurements.append_csv_row(
            (time.strftime("%Y-%m-%d %H:%M:%S"), "THD", f"{thd_percent:.4f}%", f"{thd_db:.2f} dB"),
            BASE_DIR,
        )
        self.hw_log(f"💾 Đã lưu kết quả vào '{csv_path}'.")
        artifacts = live_measurements.save_artifacts("thd", tone, recorded, fs, BASE_DIR)
        self.hw_log(f"Đã lưu TX/RX: {artifacts['tx']} | {artifacts['rx']}")
        sig_for_plot = res.get("normalized_signal", recorded)
        self._schedule_plot(self.plot_manager.open_thd_snapshot, sig_for_plot, fs, res, freq, hmax)

    def run_hw_compressor(self):
        if not self._require_sounddevice():
            return
        freq = self._parse_float(self.hw_freq, 1000.0)
        amp_max = self._parse_float(self.hw_amp, 1.36)
        in_dev, out_dev = self._device_indices()
        fs = devices.default_samplerate(out_dev or None)
        tx_meta = live_measurements.generate_compressor_tone(freq, fs, amp_max)
        tone = self._prepare_signal(tx_meta['signal'])
        self.logger.banner("Đo compressor (stepped sweep)")
        recorded = playrec.play_and_record(
            tone, fs, in_dev, out_dev, self.stop_event, log=self.hw_log, input_channels=1
        )
        if recorded is None or len(recorded) == 0:
            self.hw_log("Không ghi được dữ liệu compressor.")
            return
        self._log_recording_stats(recorded, "Compressor Rx")
        curve = live_measurements.analyze_compressor_capture(recorded, tx_meta['meta'], fs)
        if curve['no_compression']:
            self.hw_log(f"Path gain ≈ {curve['gain_offset_db']:+.2f} dB (không thấy nén)")
        else:
            self.hw_log(f"Threshold ≈ {curve['thr_db']:.2f} dBFS | Ratio ≈ {curve['ratio']:.2f}:1 | Gain offset {curve['gain_offset_db']:+.2f} dB")
        csv_row = (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "Compression",
            "No compression" if curve['no_compression'] else f"Thr {curve['thr_db']:.2f} dBFS",
            f"Ratio {curve['ratio']:.2f}:1" if not curve['no_compression'] else f"Gain {curve['gain_offset_db']:+.2f} dB",
        )
        csv_path = live_measurements.append_csv_row(csv_row, BASE_DIR)
        self.hw_log(f"💾 Đã lưu kết quả vào '{csv_path}'.")
        artifacts = live_measurements.save_artifacts("compressor", tone, recorded, fs, BASE_DIR)
        self.hw_log(f"Đã lưu TX/RX: {artifacts['tx']} | {artifacts['rx']}")
        self._schedule_plot(self.plot_manager.open_compressor_snapshot, [("Captured", curve)])

    def run_hw_attack_release(self):
        if not self._require_sounddevice():
            return
        freq = self._parse_float(self.hw_freq, 1000.0)
        amp = self._parse_float(self.hw_amp, 0.7)
        rms_win = self._parse_float(self.hw_ar_rms_win, 5)
        in_dev, out_dev = self._device_indices()
        fs = devices.default_samplerate(out_dev or None)
        tone = self._prepare_signal(attack_release.generate_step_tone(freq, fs, amp=amp))
        self.logger.banner("Đo Attack/Release")
        recorded = playrec.play_and_record(tone, fs, in_dev, out_dev, self.stop_event, log=self.hw_log)
        if recorded is None:
            self.hw_log("Không ghi được dữ liệu A/R.")
            return
        times = attack_release.attack_release_times(recorded, fs, rms_win)
        self.hw_log(f"Attack ≈ {times['attack_ms']:.1f} ms | Release ≈ {times['release_ms']:.1f} ms")
        self._schedule_plot(self.plot_manager.open_ar_snapshot, recorded, fs, rms_win, times)

    # ---------------------------------------------------------
    # LOOPBACK & FILE OPERATIONS
    # ---------------------------------------------------------
    def run_loopback_record(self):
        if not self._require_sounddevice():
            return
        in_dev, out_dev = self._device_indices()
        src = self.hw_loop_file.get()
        if not src or not os.path.isfile(src):
            messagebox.showwarning("Chọn file", "Chọn file WAV input.")
            return
        fs, data = wav_io.read_wav(src)
        if fs is None:
            self.hw_log("Không đọc được file input.")
            return
        self.state['input_file'] = src
        data = self._prepare_signal(np.asarray(data, dtype=np.float32))
        self.logger.banner("Loopback & ghi received.wav")
        recorded = playrec.play_and_record(data, fs, in_dev, out_dev, self.stop_event, log=self.hw_log)
        if recorded is None:
            self.hw_log("Loopback thất bại.")
            return
        save_path = os.path.join(BASE_DIR, "received.wav")
        if wav_io.write_wav(save_path, recorded, fs):
            self.state['received_file'] = save_path
            self.hw_log(f"Đã lưu file thu: {save_path}")
        else:
            self.hw_log("Không thể lưu received.wav")

    # ---------------------------------------------------------
    # ANALYSIS HELPERS
    # ---------------------------------------------------------
    def _analyze_single_file(self, mode: str, path: str):
        fs, data = wav_io.read_wav(path)
        if fs is None:
            self.hw_log("Không đọc được file.")
            return
        freq = self._parse_float(self.hw_freq, 1000.0)
        hmax = self._parse_int(self.hw_thd_hmax, 5)
        rms_win = self._parse_float(self.hw_ar_rms_win, 5)
        if mode == 'thd':
            res = thd.compute_thd(data, fs, freq, hmax)
            self.hw_log(f"[Single] THD {os.path.basename(path)}: {res['thd_percent']:.4f}% ({res['thd_db']:.2f} dB)")
            self._schedule_plot(self.plot_manager.open_thd_snapshot, data, fs, res, freq, hmax)
        elif mode == 'compressor':
            meta = compressor.build_stepped_tone(freq, fs)
            res = compressor.compression_curve(data, meta['meta'], fs, freq)
            if res['no_compression']:
                self.hw_log("[Single] Không phát hiện nén.")
            else:
                self.hw_log(f"[Single] Thr {res['thr_db']:.2f} dBFS | Ratio {res['ratio']:.2f}:1 | Gain {res['gain_offset_db']:+.2f} dB")
            self._schedule_plot(self.plot_manager.open_compressor_snapshot, [("Captured", res)])
        elif mode == 'ar':
            times = attack_release.attack_release_times(data, fs, rms_win)
            self.hw_log(f"[Single] Attack {times['attack_ms']:.1f} ms | Release {times['release_ms']:.1f} ms")
            self._schedule_plot(self.plot_manager.open_ar_snapshot, data, fs, rms_win, times)

    def _log_residual_metrics(self, metrics, latency_ms, gain_error_db):
        self.hw_log(f"Latency: {latency_ms:.2f} ms | Gain error: {gain_error_db:+.2f} dB")
        self.hw_log(f"Noise floor: {metrics['noise_floor_dbfs']:.2f} dBFS | Residual RMS: {metrics['residual_rms_dbfs']:.2f} dBFS")
        self.hw_log(f"SNR (est): {metrics['snr_db']:.2f} dB | THD delta: {metrics['thd_delta_db']:+.2f} dB")
        self.hw_log(f"FR deviation (median): {metrics['fr_dev_median_db']:+.2f} dB")
        hums = ", ".join([f"{h['freq']}Hz:{h['level_db']:.1f}dB" for h in metrics['hum_peaks']])
        self.hw_log(f"Hum peaks: {hums}")

    def _analyze_pair(self, mode: str, input_path: str, recv_path: str):
        freq = self._parse_float(self.hw_freq, 1000.0)
        hmax = self._parse_int(self.hw_thd_hmax, 5)
        rms_win = self._parse_float(self.hw_ar_rms_win, 5)
        fs_in, sig_in = wav_io.read_wav(input_path)
        fs_out, sig_out = wav_io.read_wav(recv_path)
        if fs_in is None or fs_out is None:
            self.hw_log("Không đọc được file input/output.")
            return
        if fs_in != fs_out:
            self.hw_log("Fs không khớp giữa hai file.")
            return
        sig_in = np.asarray(sig_in, dtype=np.float32)
        sig_out = np.asarray(sig_out, dtype=np.float32)
        max_lag = int(fs_in * 5)  # cap search to ~5s to avoid runaway correlation
        a_in, a_out, lag = compare.align_signals(sig_in, sig_out, max_lag_samples=max_lag)
        a_out, gain_err = compare.gain_match(a_in, a_out)
        latency_ms = lag / fs_in * 1000.0
        metrics = compare.residual_metrics(a_in, a_out, fs_in, freq, hmax)
        self._log_residual_metrics(metrics, latency_ms, gain_err)
        if mode == 'thd':
            self.hw_log(f"THD input: {metrics['thd_ref_db']:.2f} dB | received: {metrics['thd_tgt_db']:.2f} dB")
            res_out = thd.compute_thd(a_out, fs_in, freq, hmax)
            self._schedule_plot(self.plot_manager.open_thd_snapshot, a_out, fs_in, res_out, freq, hmax)
        elif mode == 'compressor':
            meta = compressor.build_stepped_tone(freq, fs_in)
            base_curve = compressor.compression_curve(a_in, meta['meta'], fs_in, freq)
            out_curve = compressor.compression_curve(a_out, meta['meta'], fs_in, freq)
            self.hw_log(f"Input Thr {base_curve['thr_db']:.2f} | Ratio {base_curve['ratio']:.2f}")
            self.hw_log(f"Received Thr {out_curve['thr_db']:.2f} | Ratio {out_curve['ratio']:.2f}")
            self.hw_log(f"ΔThr {out_curve['thr_db'] - base_curve['thr_db']:+.2f} dB | ΔRatio {out_curve['ratio'] - base_curve['ratio']:+.2f}")
            self._schedule_plot(self.plot_manager.open_compressor_snapshot, [("Input", base_curve), ("Output", out_curve)])
        elif mode == 'ar':
            cmp_ar = attack_release.compare_attack_release(a_in, a_out, fs_in, rms_win)
            self.hw_log(f"Attack in/out: {cmp_ar['input']['attack_ms']:.1f} / {cmp_ar['output']['attack_ms']:.1f} ms | Δ {cmp_ar['delta_attack']:+.1f} ms")
            self.hw_log(f"Release in/out: {cmp_ar['input']['release_ms']:.1f} / {cmp_ar['output']['release_ms']:.1f} ms | Δ {cmp_ar['delta_release']:+.1f} ms")
            self._schedule_plot(self.plot_manager.open_ar_snapshot, a_out, fs_in, rms_win, cmp_ar['output'])

    def analyze_loopback(self, mode: str):
        inp = self.state.get('input_file') or self.hw_loop_file.get()
        rec = self.state.get('received_file')
        if inp and rec and os.path.isfile(inp) and os.path.isfile(rec):
            self.logger.banner("COMPARE MODE (Loopback)")
            self._analyze_pair(mode, inp, rec)
        elif inp and os.path.isfile(inp):
            self.logger.banner("SINGLE FILE MODE (Loopback)")
            self._analyze_single_file(mode, inp)
        else:
            self.hw_log("Chưa có file loopback để phân tích.")

    def analyze_offline(self, mode: str):
        inp = self.offline_in.get()
        out = self.offline_out.get()
        if inp and out and os.path.isfile(inp) and os.path.isfile(out):
            self.logger.banner("COMPARE MODE (Offline)")
            self._analyze_pair(mode, inp, out)
        elif inp and os.path.isfile(inp):
            self.logger.banner("SINGLE FILE MODE (Offline)")
            self._analyze_single_file(mode, inp)
        elif out and os.path.isfile(out):
            self.logger.banner("SINGLE FILE MODE (Offline out)")
            self._analyze_single_file(mode, out)
        else:
            self.hw_log("Chọn ít nhất một file để phân tích.")

    # ---------------------------------------------------------
    def _now_str(self):
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalysisToolkitApp(root)
    root.mainloop()
