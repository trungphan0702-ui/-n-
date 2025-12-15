Audio Measurement Toolkit

(Tkinter GUI – Backend Contract–First Architecture)

1. Project Overview

Purpose
Toolkit đo lường & phân tích hệ thống audio, sử dụng GUI Tkinter làm lớp điều khiển, backend đảm nhiệm toàn bộ DSP, audio I/O và xuất kết quả.

Các nhóm đo chính (bắt buộc cho mọi mode):

THD (Total Harmonic Distortion)

Compressor characteristics (threshold, ratio, gain offset, curve)

Attack / Release (time constants)

Use cases

Realtime hardware loopback: phát stimulus → qua thiết bị ngoài → thu về → phân tích realtime + streaming plot

Offline analysis: phân tích WAV có sẵn

Compare input vs output: align, gain-match, latency, residual metrics

2. Architectural Principles (NON-NEGOTIABLE)
2.1 Contract-first (BẮT BUỘC)

backend/contracts.py là public API duy nhất

GUI chỉ import & gọi contracts

Không gọi trực tiếp analysis/, audio/, utils/

2.2 GUI bất biến

File tham chiếu: GUI_D_3_2_1.py

Không được:

Thay đổi layout

Đổi tên widget / tab / button

Tách GUI sang file khác

GUI chỉ:

Thu thập input

Gọi backend contracts

Hiển thị log / kết quả / plot

2.3 Backend không phụ thuộc GUI

Backend không import tkinter, messagebox, GUI state

Mọi xử lý dài chạy trong thread backend

3. System Architecture (Textual)
g1-main/
├─ GUI_D_3_2_1.py          ⭐ GUI chính (IMMUTABLE)
│
├─ backend/
│  └─ contracts.py        ⭐ PUBLIC API DUY NHẤT
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

4. Backend API Contract (Chuẩn bắt buộc)
4.1 Quy ước chữ ký hàm

Sync (nhanh, offline):

run_xxx(request) -> XxxResult


Async (realtime, loopback):

start_xxx(
    request,
    *,
    stop_event,
    on_progress,
    on_log
) -> XxxHandle


GUI không được tự tạo thread DSP, chỉ dùng handle.

5. Streaming Realtime (BẮT BUỘC)
5.1 ProgressEvent

Dùng cho realtime plot

Payload KHÔNG là matplotlib figure

ProgressEvent(
    phase="streaming",
    percent=None,
    message="",
    meta={
        "chunk": i,
        "data": {
            # spectrum / envelope / gain_reduction / ...
        }
    }
)


chunk bắt buộc

GUI dùng meta["data"] để vẽ realtime

6. Artifact & Metadata Standard

Mọi phép đo phải xuất artifact (CSV và/hoặc WAV).

6.1 Artifact fields
Artifact(
    kind="wav | csv | json",
    path="...",
    meta={
        "feature": "thd | compressor | attack_release | compare",
        "mode": "offline | loopback",
        "sample_rate": 48000,
        "channels": 1,
        "input_device": "...",
        "output_device": "...",
        "stimulus": {...},
        "run_id": "...",
        "timestamp": "ISO-8601"
    }
)


👉 Mục tiêu: không nhầm giữa các lần chạy / chế độ / thiết bị

7. Measurement Coverage Rules
FeatureOfflineRealtime
THD✅✅
Compressor✅✅
Attack/Release✅✅

Không được có feature “chỉ offline” hoặc “chỉ realtime”.

8. Execution Model
8.1 Realtime loopback

GUI → start_xxx(...)

Backend:

validate device

generate stimulus

play & record

stream chunk → on_progress

phân tích DSP

export artifact

GUI:

hiển thị log

vẽ realtime

chờ handle.join()

8.2 Offline

GUI chọn WAV

GUI gọi run_xxx(...)

Backend:

đọc WAV

phân tích

trả summary + plots + artifacts

9. Threading & Cancellation

Mọi realtime task:

chạy trong backend thread

bắt buộc check stop_event

GUI chỉ gọi:

handle.cancel()

handle.join()

10. Plotting Strategy

Backend chỉ trả PlotSpec

GUI / utils:

translate PlotSpec → matplotlib

Không tạo plot trong backend

11. Testing Strategy

tests/self_test.py:

test DSP offline

test I/O cơ bản

Không test GUI tự động (GUI immutable)

12. Extension Rules (RẤT QUAN TRỌNG)
Được phép

Thêm phép đo mới → thêm API trong contracts.py

Mở rộng DSP trong analysis/

Thêm field vào summary / artifact / meta

CẤM

Đưa DSP vào GUI

GUI import trực tiếp analysis/, audio/

Thay đổi layout GUI

Bỏ qua streaming chunk hoặc artifact metadata

13. Final Statement

Backend public API = backend/contracts.py
GUI chỉ là client của contracts.
Kiến trúc này đã CHỐT và là nền tảng cho toàn bộ phát triển tiếp theo
