# Audio Measurement Toolkit
(Tkinter GUI – Backend Contract–First Architecture)

============================================================

1. PROJECT OVERVIEW
-------------------

Purpose:
- Toolkit đo lường & phân tích hệ thống audio, sử dụng GUI Tkinter làm lớp điều khiển.
- Backend chịu trách nhiệm toàn bộ DSP, audio I/O, streaming realtime và xuất artifact.

Các nhóm phép đo bắt buộc:
- THD (Total Harmonic Distortion)
- Compressor characteristics (threshold, ratio, gain offset, curve)
- Attack / Release (time constants)

Use cases:
- Realtime hardware loopback: phát stimulus → qua thiết bị ngoài → thu về → phân tích realtime
- Offline analysis: phân tích WAV có sẵn
- Compare input vs output: align, gain-match, latency, residual metrics


2. ARCHITECTURAL PRINCIPLES (NON-NEGOTIABLE)
--------------------------------------------

2.1 Contract-first (BẮT BUỘC)
- backend/contracts.py là public API duy nhất
- GUI chỉ import và gọi functions từ backend/contracts.py
- GUI tuyệt đối không import trực tiếp analysis/, audio/, utils/

2.2 GUI bất biến
- File tham chiếu: GUI_D_3_2_1.py
- Không được:
  - Thay đổi layout
  - Đổi tên widget / tab / button
  - Di chuyển widget
  - Tách GUI sang file khác
- GUI chỉ:
  - Thu thập input
  - Gọi backend contracts
  - Hiển thị log / kết quả / plot

2.3 Backend độc lập GUI
- Backend không import tkinter, messagebox, Tk variables
- Backend không phụ thuộc GUI state
- Mọi xử lý dài chạy trong backend thread


3. SYSTEM ARCHITECTURE
----------------------

g1-main/
├─ GUI_D_3_2_1.py          (IMMUTABLE GUI)
│
├─ backend/
│  └─ contracts.py        (PUBLIC API DUY NHẤT)
│
├─ analysis/
│  ├─ thd.py
│  ├─ compressor.py
│  ├─ attack_release.py
│  ├─ compare.py
│  └─ live_measurements.py
│
├─ audio/
│  ├─ devices.py
│  ├─ playrec.py
│  └─ wav_io.py
│
├─ utils/
│  ├─ threading.py
│  ├─ logging.py
│  └─ plot_windows.py
│
├─ tests/
│  └─ self_test.py
└─ requirements.txt


4. BACKEND API CONTRACT STANDARD
--------------------------------

4.1 Function naming rules

Sync (nhanh, offline):
- run_xxx(request) -> XxxResult

Async (realtime, loopback):
- start_xxx(request, *, stop_event, on_progress, on_log) -> XxxHandle

GUI không được tự tạo thread DSP.


5. STREAMING REALTIME (BẮT BUỘC)
--------------------------------

- Mọi realtime measurement phải stream dữ liệu theo chunk
- Dùng ProgressEvent
- Không trả matplotlib figure

Chuẩn ProgressEvent:
- phase = "streaming"
- meta bắt buộc có:
  - "chunk": index
  - "data": payload thuần (spectrum / envelope / gain_reduction / ...)

Ví dụ:
ProgressEvent(
    phase="streaming",
    meta={
        "chunk": i,
        "data": {...}
    }
)


6. ARTIFACT & METADATA STANDARD
-------------------------------

Mọi phép đo phải xuất artifact CSV và/hoặc WAV.

Artifact meta bắt buộc tối thiểu:
- feature: thd | compressor | attack_release | compare
- mode: offline | loopback
- sample_rate
- channels
- input_device (nếu loopback)
- output_device (nếu loopback)
- stimulus (freq, level, step, v.v.)
- run_id (UUID)
- timestamp (ISO-8601)

Ví dụ:
Artifact(
    kind="wav",
    path="...",
    meta={
        "feature": "thd",
        "mode": "loopback",
        "sample_rate": 48000,
        "channels": 1,
        "input_device": "...",
        "output_device": "...",
        "stimulus": {...},
        "run_id": "...",
        "timestamp": "..."
    }
)


7. FEATURE COVERAGE RULE
------------------------

Mọi feature phải có cả Offline và Realtime:

- THD: offline + loopback
- Compressor: offline + loopback
- Attack/Release: offline + loopback

Không được tồn tại feature chỉ chạy 1 mode.


8. EXECUTION MODEL
------------------

8.1 Realtime loopback
- GUI gọi start_xxx()
- Backend:
  - validate device
  - generate stimulus
  - play & record
  - stream chunk qua on_progress
  - phân tích DSP
  - export artifact
- GUI:
  - hiển thị log
  - vẽ realtime
  - gọi handle.join()

8.2 Offline
- GUI chọn WAV
- GUI gọi run_xxx()
- Backend:
  - đọc WAV
  - phân tích
  - trả summary + plots + artifacts


9. THREADING & CANCELLATION
---------------------------

- Mọi realtime task phải:
  - chạy trong backend thread
  - check stop_event thường xuyên
- GUI chỉ:
  - handle.cancel()
  - handle.join()


10. PLOTTING STRATEGY
---------------------

- Backend chỉ trả PlotSpec (data + metadata)
- GUI hoặc utils chịu trách nhiệm render matplotlib
- Backend tuyệt đối không tạo plot


11. TESTING STRATEGY
--------------------

- tests/self_test.py:
  - test DSP offline
  - test audio I/O
- GUI testing: manual (GUI immutable)


12. EXTENSION RULES (RẤT QUAN TRỌNG)
------------------------------------

ĐƯỢC PHÉP:
- Thêm feature mới qua backend/contracts.py
- Mở rộng analysis/, audio/
- Thêm field vào summary / artifact / meta

CẤM:
- Đưa DSP vào GUI
- GUI import analysis/audio trực tiếp
- Thay đổi layout GUI
- Bỏ qua streaming chunk hoặc metadata


13. FINAL STATEMENT
-------------------

Backend public API = backend/contracts.py

GUI chỉ là client của contracts.

Tài liệu này là SOURCE OF TRUTH cho toàn bộ project.
Mọi code viết sau phải tuân thủ tuyệt đối tài liệu này.
