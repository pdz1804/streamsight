# Inference guide

Where each step runs, and what it costs in memory.

## Prerequisites

| | Required | Notes |
|---|---|---|
| Python | 3.11 | |
| Node | 20+ | 22 LTS recommended |
| GPU | optional | NVIDIA with a CUDA 12 driver. Without one, everything runs on the CPU path |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# torch first, from the CUDA 12.1 index
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python ml/scripts/fetch_assets.py
```

`fetch_assets.py` downloads YOLO11n weights (5.35 MB), the demo clip, and the held-out calibration
clip. None of these are committed: they are large, reproducible, and third-party.

Optionally build the alternative backends:

```powershell
python ml/quantization/export_engines.py --formats fp16_onnx openvino_cpu
python ml/quantization/calibrate.py --frames 128
```

## Running

```powershell
# API, :8100
cd apps/api
python -m uvicorn app.main:app --port 8100

# Web console, :3100
cd apps/web
npm install
npm run dev
```

The API boots even when no backend can load, and explains the problem through `/health` and a 503 on
inference routes. A service that refuses to start is much harder to diagnose than one that says what
is wrong.

## Backend selection

At startup the runtime walks a ladder, cheapest-capable-first, and takes the first backend that both
loads **and** survives a warmup inference:

```
int8_trt -> fp16_trt -> fp16_onnx -> fp32_gpu -> openvino_cpu -> int8_onnx_cpu -> fp32_cpu
```

Override with `STREAMSIGHT_DEFAULT_PRECISION=openvino_cpu`, or change it live at
`/settings` in the UI.

## Memory budget

Measured on an RTX A1000 Laptop GPU (4096 MiB) at 640 px:

| Stage | VRAM |
|---|---|
| Model + CUDA context after load | ~131 MiB |
| Peak during a 200-frame run | **316 MiB** |
| Budget | 3500 MiB |
| Headroom | ~91% |

Frame buffers live in host RAM, not VRAM: the ring buffer holds 30 decoded frames, roughly 80 MB at
1280x720.

### Resolution policy

At startup the runtime reads **free VRAM immediately after the model and its runtime workspace are
resident**, before any frame buffers exist. Below 1200 MiB free it starts at 480 px instead of 640
and flags degraded mode. On this hardware free-after-load is ~3.6 GB, so it stays at 640.

## Configuration

All settings take a `STREAMSIGHT_` prefix and can go in a `.env` at the repo root.

| Setting | Default | Purpose |
|---|---|---|
| `DEFAULT_PRECISION` | `auto` | Preferred backend, or `auto` for the ladder |
| `DEFAULT_IMGSZ` | `640` | Inference resolution |
| `DEGRADED_IMGSZ` | `480` | Resolution used after a VRAM event |
| `CONF_THRESHOLD` | `0.25` | Detection confidence floor |
| `IOU_THRESHOLD` | `0.45` | NMS IoU |
| `VRAM_FREE_FLOOR_MB` | `1200` | Below this free-after-load, start at `DEGRADED_IMGSZ` |
| `CAPTURE_MAX_WIDTH` | `1280` | Cap decoded frame width; `0` disables |
| `STREAM_MAX_WIDTH` | `1280` | Cap streamed frame width |
| `JPEG_QUALITY` | `80` | Streamed frame quality |
| `RING_BUFFER_SIZE` | `30` | Capture buffer depth |

## Sources

- **File** upload in the UI, or any path the API can read
- **Webcam** device index, e.g. `0`
- **RTSP** any `rtsp://` URL

Files are paced to their native frame rate so playback runs at real speed. Live sources are never
paced and drop the oldest buffered frame under load.

## Verifying the fallback path

`/settings` has a **Trigger degradation** button, or:

```powershell
curl -X POST http://localhost:8100/config/degrade
```

One step down the ladder, `degraded_mode: true`, and a reason string. Worth running once so the
behaviour is observed rather than assumed.

## Troubleshooting

**A backend shows "failed on this host"** - it loaded but could not execute. The recorded reason is
shown in the model selector. Common cause is ONNX Runtime needing a cuDNN version the installed CUDA
toolkit does not ship.

**"was exported at 640 px and cannot run at 480 px"** - exported graphs have fixed input shapes.
Re-export at the target resolution, or use a PyTorch backend, which accepts any size.

**Viewer FPS well below the benchmark** - expected. The benchmark measures detect and track; the
viewer additionally annotates, encodes and transports. See [BENCHMARKS.md](BENCHMARKS.md).

**Everything is slow and VRAM looks full** - something else is using the GPU. On a 4 GB card,
sharing it with another process dominates every other cost.

Phu Nguyen - HCMC, Vietnam
