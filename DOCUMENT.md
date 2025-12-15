# Audio Measurement Toolkit (Tkinter GUI)

## 1. Project Overview
- **Purpose:** Tkinter-based toolkit for measuring and analyzing audio systems via total harmonic distortion (THD), compressor characteristics, and attack/release timing.
- **Use cases:**
  - **Hardware measurement:** Real-time playback/record loopback against a soundcard for THD, compressor response, and time-constant profiling.
  - **Offline analysis:** Process existing WAV captures to extract DSP metrics without hardware.
  - **Input vs Output comparison:** Evaluate how output deviates from reference signals across level-dependent or time-varying processing.

## 2. Architectural Principles
- **Separation of concerns:** GUI handles presentation and high-level orchestration only. DSP, I/O, and utility concerns reside strictly under `analysis/`, `audio/`, and `utils/`.
- **Immutable GUI contract:** `GUI_D_3_2_1.py` is the reference GUI. Do **not** change layout, move widgets, rename buttons/tabs/elements, split or refactor GUI into other files. All logic must remain in backend modules; GUI only wires user actions to backend calls.
- **Backend-driven design:** New features are added by extending backend modules and invoking them from existing GUI hooks. GUI never contains DSP or device-specific code.

## 3. System Architecture Diagram (Textual)
```
g1-main/
├─ GUI_D_3_2_1.py          ⭐ GUI chính (REFERENCE – KHÔNG ĐƯỢC ĐỘNG)
├─ analysis/
│  ├─ __init__.py
│  ├─ thd.py               # Tính THD, harmonic
│  ├─ compressor.py        # Phân tích compressor curve
│  ├─ attack_release.py    # Attack / Release
│  ├─ compare.py           # Align, gain-match, residual
│  └─ live_measurements.py # Orchestrator cho đo realtime
│
├─ audio/
│  ├─ __init__.py
│  ├─ devices.py           # Liệt kê & parse soundcard
│  ├─ playrec.py           # play & record realtime
│  └─ wav_io.py            # read/write wav
│
├─ utils/
│  ├─ __init__.py
│  ├─ logging.py           # UILogger
│  ├─ threading.py         # run_in_thread + stop_event
│  └─ plot_windows.py      # Cửa sổ plot (Tk / matplotlib)
│
├─ tests/
│  └─ self_test.py         # offline DSP self-test
│
├─ requirements.txt
└─ .gitignore
```

## 4. Module Responsibilities
### 4.1 GUI Layer (`GUI_D_3_2_1.py`)
- **Allowed:** Collect user parameters, trigger backend functions, start/stop background threads, show log/status text, open/close plot windows via utilities.
- **Forbidden:** DSP algorithms, audio device calls, WAV parsing, plotting logic, or any blocking work. Must not alter layout or widget identities; must not relocate GUI code.

### 4.2 Analysis Layer (`analysis/`)
- `thd.py`: Compute THD metrics from captured or offline signals; prepare data for plotting (harmonic spectra, level curves).
- `compressor.py`: Derive threshold/ratio/curve characteristics; map input-output levels and gain reduction behavior.
- `attack_release.py`: Measure attack and release time constants using envelope tracking on transient stimuli.
- `compare.py`: Align and compare input vs output signals; produce deviation/latency metrics and overlays.
- `live_measurements.py`: Coordinate real-time measurement flows (loopback scheduling, stimulus generation, streaming callbacks) without GUI knowledge.

### 4.3 Audio Layer (`audio/`)
- `devices.py`: Enumerate/select soundcard devices and channel configurations through a hardware abstraction interface.
- `playrec.py`: Manage synchronized playback/record streams for loopback measurements (buffering, latency handling).
- `wav_io.py`: Read/write WAV files for offline analysis or exporting captures.

### 4.4 Utilities (`utils/`)
- `threading.py`: Thread helpers to launch and control background tasks with cooperative cancellation (e.g., `stop_event`).
- `logging.py`: Centralized logging hooks for GUI-safe status updates and backend diagnostics.
- `plot_windows.py`: Manage lifecycle of matplotlib windows independent of Tkinter main loop; ensure non-blocking visualization.

## 5. Execution Flows
- **Realtime THD measurement:**
  1. GUI collects device/level settings and calls `analysis.live_measurements` helpers.
  2. `audio.playrec` starts loopback playback/record; raw buffers stream to analysis.
  3. `analysis.thd` computes harmonic metrics; `utils.plot_windows` renders spectra/plots.
  4. GUI receives updates/logs and renders statuses without blocking.

- **Offline WAV analysis:**
  1. GUI prompts for WAV paths and invokes `audio.wav_io` to load reference/measurement files.
  2. Appropriate analysis module (`thd`, `compressor`, `attack_release`, or `compare`) processes arrays.
  3. Results routed to plot windows and GUI log area.

- **Input vs Output comparison:**
  1. GUI gathers reference/output selections.
  2. `audio.wav_io` loads files or `audio.playrec` streams live signals.
  3. `analysis.compare` aligns signals, computes deltas/latency, and provides overlays.
  4. Plots rendered via `plot_windows`; GUI shows summary values.

## 6. Threading and Responsiveness Model
- Background measurements/analysis run via `utils.threading` helpers, using worker threads to keep Tkinter responsive.
- `stop_event` (or equivalent flag) enables cooperative cancellation for long-running streams/analyses, checked within audio and analysis loops.
- GUI callbacks only start/stop threads and handle UI state; they never block the Tkinter main loop.

## 7. Plotting Strategy
- Plot windows are decoupled from Tkinter to avoid blocking; `utils.plot_windows` opens standalone matplotlib windows for each result set.
- Lifecycles: created per run, closed via utility helpers or user action; GUI should request closures but never manage matplotlib internals directly.

## 8. Testing Strategy
- `tests/self_test.py` performs automated checks of DSP routines and I/O helpers without involving GUI interactions.
- Offline DSP testing validates numeric correctness and file handling; GUI testing is manual or exploratory, since GUI layout is immutable and non-negotiable.

## 9. Extension Guidelines (IMPORTANT)
- **Adding measurements:** Implement DSP in a new module under `analysis/` (or extend existing ones) and expose callable functions; update GUI callbacks to call these functions without altering layout.
- **Adding DSP metrics:** Extend computation in analysis modules; return additional values/plots via existing plot/log interfaces. GUI may display extra text but must not change widget structure.
- **Backend-only changes:** Use `audio/` for device/WAV work and `utils/` for threading/logging/plot helpers. Do **not** embed new logic in GUI; only wire new backend functions to existing buttons/menu actions.
- **Preserve architectural rules:** `GUI_D_3_2_1.py` stays the single GUI file and must remain presentation/orchestration only.

## 10. Anti-Patterns (Explicit Warnings)
- Placing DSP or audio code inside `GUI_D_3_2_1.py`.
- Calling sounddevice or low-level audio APIs directly from the GUI.
- Blocking Tkinter main loop with long computations or sleeps.
- Changing GUI layout, moving widgets, renaming controls, or splitting GUI code across files.
- Bypassing `utils.plot_windows` for visualization or creating blocking plots.
- Ignoring `stop_event`/cancellation patterns in long-running tasks.
