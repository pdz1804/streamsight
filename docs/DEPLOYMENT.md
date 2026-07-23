# Deployment

**LOCAL.** Everything on this page runs on one machine. There is no cloud component, no managed
service, and no cost. GPU is optional throughout — the CPU path is a real target, not a fallback.

---

## Ports

| Port | Service | Notes |
|---|---|---|
| 8100 | FastAPI API | REST + the `/detect/stream` WebSocket |
| 3100 | Next.js console | Talks to 8100 from the **browser**, not server-side |

Both are fixed defaults. The console resolves the API from `NEXT_PUBLIC_API_BASE`, which Next
inlines **at build time** (`next.config.mjs`), so changing it after a build has no effect — rebuild.

The API's CORS allowlist defaults to `http://localhost:3100` and `http://127.0.0.1:3100`. Serving
the console from any other origin means setting `STREAMSIGHT_CORS_ORIGINS` too.

---

## Local run

Requires Python 3.11 and Node 20+ (CI uses Node 22).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# GPU machine (CUDA 12 driver line):
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
# CPU-only machine: same command with /cpu instead of /cu121

pip install -r requirements.txt
python ml/scripts/fetch_assets.py          # weights + demo clip + calibration clip
```

Two shells:

```powershell
python -m uvicorn app.main:app --app-dir apps/api --port 8100
```

```powershell
cd apps/web
npm install
npm run dev
```

Open <http://localhost:3100>.

### Optional: the faster CPU backend

`fp32_cpu` runs straight off the downloaded `.pt` weights and needs no export. The measured-fastest
CPU backend, OpenVINO, does:

```powershell
python ml/quantization/export_engines.py --formats fp16_onnx openvino_cpu
python ml/quantization/calibrate.py        # INT8 ONNX, held-out calibration clip
```

### Makefile

A `Makefile` wraps all of the above. It is **optional convenience only** — Windows is the primary
shell here and does not ship `make`, so every target is a one-line wrapper around a command written
out in this file or in the README. `make help` lists them: `setup`, `api`, `web`, `test`,
`test-fast`, `lint`, `fmt`, `bench`, `frontier`, `soak`, `export`.

---

## Docker run

```powershell
docker compose -f infra/docker-compose.yml up --build
```

Then <http://localhost:3100>. Two images are built from the repo root as context:

| File | Base | Contents |
|---|---|---|
| `infra/Dockerfile.api` | `python:3.11-slim` | API + `ml/`, torch from the **CPU** wheel index |
| `infra/Dockerfile.web` | `node:22-alpine` | Next.js production build, dev deps pruned at runtime |

The API image runs `ml/scripts/fetch_assets.py` during the build, so it needs network access then
and ships with the weights and the demo clip baked in (~35 MB). The SQLite detection log and
uploaded clips live in the `api-data` named volume and survive a rebuild.

### The CPU-only caveat

**There is no CUDA base image and `docker compose up` gives you no GPU inference.** That is
deliberate, and it is the honest portable path (NFR-7): the container has no host driver coupling
and runs identically on any x86-64 Docker host.

The cost is throughput. Measured on this machine, natively:

- FP32 PyTorch **GPU** @ 640 px — 48.5 FPS
- OpenVINO **CPU** @ 640 px — 35.8 FPS
- FP32 PyTorch **CPU** @ 640 px — 15.4 FPS

The container defaults to `STREAMSIGHT_DEFAULT_PRECISION=auto`, which with no GPU present lands on
`fp32_cpu` — the slowest of the three, because it is the only one that needs no export step. To get
the OpenVINO number instead, export the artifact on the host and mount it:

```powershell
python ml/quantization/export_engines.py --formats openvino_cpu
# then add to the api service in infra/docker-compose.yml:
#   volumes:
#     - ../ml/models:/app/ml/models
#   environment:
#     STREAMSIGHT_DEFAULT_PRECISION: openvino_cpu
```

Those container figures have **not** been measured — the numbers above are native. Docker on Windows
adds a VM hop, so treat them as an upper bound until you benchmark inside the container with
`python ml/eval/benchmark_inference.py --engine fp32_cpu --frames 200`.

Running GPU inference in a container is possible but is not configured here. It needs the NVIDIA
Container Toolkit on the host, a CUDA runtime base image, `deploy.resources.reservations.devices`
(or `--gpus all`), and torch reinstalled from the cu121 index. None of that is tested in this repo.

---

## Environment variables

All API settings use the `STREAMSIGHT_` prefix and are read by `apps/api/app/core/config.py`. Copy
`.env.example` to `.env` at the repo root, or set them in the compose file. Every one is optional.

| Variable | Default | Purpose |
|---|---|---|
| `STREAMSIGHT_DEFAULT_PRECISION` | `auto` | `auto` walks the fallback ladder; or name a backend (`fp32_gpu`, `openvino_cpu`, `int8_onnx_cpu`, `fp32_cpu`, …) |
| `STREAMSIGHT_DEFAULT_IMGSZ` | `640` | Inference resolution |
| `STREAMSIGHT_DEGRADED_IMGSZ` | `480` | Resolution after an OOM degrade |
| `STREAMSIGHT_CONF_THRESHOLD` | `0.25` | Detection confidence floor |
| `STREAMSIGHT_IOU_THRESHOLD` | `0.45` | NMS IoU |
| `STREAMSIGHT_VRAM_FREE_FLOOR_MB` | `1200` | Free VRAM after load below which the pipeline starts degraded |
| `STREAMSIGHT_VRAM_BUDGET_MB` | `3500` | Declared budget, reported in `/metrics` |
| `STREAMSIGHT_CAPTURE_MAX_WIDTH` | `1280` | Downscale cap on the capture thread; `0` disables |
| `STREAMSIGHT_STREAM_MAX_WIDTH` | `1280` | Cap on what reaches the browser |
| `STREAMSIGHT_JPEG_QUALITY` | `80` | WebSocket frame encode quality |
| `STREAMSIGHT_RING_BUFFER_SIZE` | `30` | Drop-oldest frame queue depth |
| `STREAMSIGHT_API_PORT` | `8100` | Reported port (uvicorn's `--port` is what actually binds) |
| `STREAMSIGHT_MAX_UPLOAD_BYTES` | `33554432` | Upload ceiling, 32 MiB |
| `STREAMSIGHT_CORS_ORIGINS` | `["http://localhost:3100","http://127.0.0.1:3100"]` | Allowed browser origins |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8100` | Console → API base URL, **inlined at build time** |

---

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

- **api** — Python 3.11, CPU torch, `ruff check`, `black --check`, and `pytest apps/api/tests -q -m
  "not slow"`. The `slow` tier boots the real model and is deselected because hosted runners have
  neither a GPU nor model weights.
- **web** — Node 22, `npm ci`, `npm run typecheck`, `npm run build`.

Neither job needs a GPU, model weights, or a running service. The browser E2E suite
(`apps/web/e2e`, Playwright) is **not** in CI — it drives a live API against real video and belongs
on a machine with weights.

---

## Security posture

There is no authentication. `POST /config/model` and `POST /config/degrade` are unauthenticated, the
upload endpoint accepts video from any allowed origin, and the WebSocket is open. This is a
single-operator local demo — do not bind it to an interface you do not control, and do not put the
compose stack on a public host without putting a reverse proxy and auth in front of it.

Phu Nguyen - HCMC, Vietnam
