# StreamSight — technical deep dive

A complete technical reference for this project: what was built, why each decision was made, what
was measured, and what went wrong along the way. Written to be usable as CV and interview material,
so the emphasis is on *reasoning and evidence* rather than feature lists.

---

## 1. One-paragraph summary

StreamSight is a real-time multi-object tracking system that runs inside a 4 GB laptop GPU. It
detects and tracks objects in video from a file, webcam or RTSP feed, streams annotated frames to a
browser console over a WebSocket, and reports live throughput, latency percentiles and VRAM. It ships
with a quantization and export pipeline (ONNX FP16, OpenVINO, static-QDQ INT8) and an
accuracy-throughput frontier that scores every backend against an FP32 baseline, so the choice of
deployment target is measured rather than assumed.

**Headline numbers on an RTX A1000 Laptop GPU (4096 MiB):** 48.5 FPS at 640 px, 316 MiB peak VRAM
against a 3.5 GB budget, 12 ms end-to-end browser latency.

---

## 2. Capability matrix

| Capability | Implementation | Evidence |
|---|---|---|
| Object detection | YOLO11n, 2.6 M params, COCO-pretrained | 48.5 FPS @ 640 px |
| Multi-object tracking | ByteTrack via Ultralytics `model.track(persist=True)` | 176 unique identities over a 200-frame clip |
| Real-time streaming | WebSocket, server-side annotation, base64 JPEG | 13.4 FPS viewer, ~12 ms send-to-paint |
| Runtime model swap | Locked rebuild of the detector, no restart | Verified live via the settings UI |
| Graceful degradation | 640 -> 480 px -> cheaper backend -> CPU | Exercisable via `POST /config/degrade` |
| Quantization | ONNX Runtime static QDQ with held-out calibration | 4.36 MB artifact, 93.3% recall |
| Edge export | ONNX FP16, OpenVINO IR | 5.14 MB / 5.45 MB |
| Frontier analysis | Backend x resolution sweep, post-NMS agreement scoring | `ml/eval/reports/frontier.{md,json,png}` |
| Observability | Rolling FPS, p50/p95 latency, VRAM, CPU, RAM, track counts | `/metrics`, 1 Hz dashboard |
| Persistence | Async SQLite writer for frame summaries and track lifecycles | Bounded queue, drops under load |

---

## 3. Architecture

```
video source ──> capture thread ──> ring buffer ──> inference runtime ──> annotate ──> JPEG ──> WebSocket ──> canvas
 file/webcam/     downscale to       drop-oldest     YOLO11n + ByteTrack     burn in     encode     :8100        :3100
 RTSP             1280 px            depth 30        backend ladder          overlay
```

### Backend layering (`apps/api/app/`)

| Module | Responsibility |
|---|---|
| `main.py` | App factory, lifespan, domain-error handling |
| `config.py` | Settings, GPU probe, startup resolution policy |
| `backends.py` | Declarative registry of runnable artifacts + fallback ladder |
| `runtime.py` | Sole owner of the loaded model: selection, hot-swap, degradation |
| `detector.py` | Ultralytics wrapper, OOM classification, warmup |
| `tracker.py` | ByteTrack config generation, `Results` -> API schemas |
| `capture.py` | Threaded source, bounded ring buffer, pacing, downscale |
| `annotate.py` | Overlay drawing with per-identity stable colours |
| `streaming.py` | WebSocket session, blocking work off the event loop |
| `metrics.py` | Bounded rolling telemetry |
| `store.py` | Async SQLite writer |
| `models.py` | Pydantic contract, mirrored in `apps/web/lib/types.ts` |
| `routers/` | HTTP + WebSocket endpoints |

The separation that matters most: **one object owns the model**. Everything that constructs, swaps
or destroys a detector goes through `InferenceRuntime` under a single lock. That is what makes
changing precision in the middle of a live stream safe rather than a race.

---

## 4. Design decisions worth defending

### 4.1 Detect and track in one call

Ultralytics exposes `model.track(frame, tracker="bytetrack.yaml", persist=True)`, which runs
detection and association together and carries tracker state across calls. The alternative — a
standalone ByteTrack package fed `[N,6]` arrays — means maintaining a second dependency and a
coordinate contract for no gain. There is no such Ultralytics API as a plain
`update([N,6]) -> [N,7]`; the tracker state lives on the predictor.

### 4.2 A declarative backend registry

Seven backends (`int8_trt`, `fp16_trt`, `fp16_onnx`, `fp32_gpu`, `openvino_cpu`, `int8_onnx_cpu`,
`fp32_cpu`) are described in one table: artifact path, device, GPU requirement, whether the input
shape is fixed. The API, the model-selector UI, the benchmark harness and the export scripts all
read that table. Duplicating path logic across four consumers is how they silently disagree about
what `int8_trt` means.

### 4.3 Warmup is part of backend selection

**A backend that loads is not a backend that works.** ONNX Runtime advertises a CUDA execution
provider whenever the GPU build is installed, then fails at the first tensor bind if cuDNN is the
wrong major version. So selection loads *and* warms up, and a warmup failure rejects the backend
exactly like a load failure.

Failures are remembered as `(backend, resolution)` pairs, not bare backend names. Exported graphs
bake in their input resolution, so a 640 px ONNX genuinely cannot serve 480 px — blacklisting the
whole backend for that would discard a working configuration. The recorded reason is surfaced in the
UI, so the interface never offers an option that dies when clicked.

### 4.4 Degradation you can trigger

On CUDA OOM: drop to 480 px if above it, else step to a cheaper backend, else report unrecoverable.
Every step sets `degraded_mode` with a human-readable reason.

`POST /config/degrade` runs one step on demand. This is a deliberate product decision: a fallback
path that is never exercised is a fallback nobody can trust. In testing it revealed correct
emergent behaviour — degrading from `openvino_cpu@640` dropped to 480 px, found OpenVINO could not
serve that shape, and fell through to the dynamic-shape PyTorch CPU backend.

### 4.5 Server-side overlays

Boxes are burned into the JPEG rather than composited client-side, so pixels and overlay cannot
desynchronise while frames are in flight. The client still receives structured tracks for the
legend, and `apps/web/lib/palette.ts` mirrors `apps/api/app/annotate.py` exactly so a legend swatch
matches its box. Colours are keyed on track id, so an object keeps its colour as long as it keeps
its identity — that visual continuity is what makes tracking legible to a human.

### 4.6 Drop-oldest, and pace files

The capture buffer is bounded and drops the **oldest** frame. For a live camera, a fresh frame
delivered late beats a stale frame delivered on time. Recorded files are additionally paced to their
native frame rate, so a clip plays at real speed instead of being drained as fast as the disk allows.

### 4.7 Bounded telemetry

Fixed-length deques for time series, a capped set for unique ids. A multi-hour stream must not grow
the collector — an unbounded metrics buffer would be the first thing to break a no-leak requirement.
FPS is computed from frame **arrival times**, not `1000 / latency`, so queueing and encode cost are
included. That is the rate a viewer actually perceives.

### 4.8 Lossy telemetry persistence

Writes go through a bounded queue to a dedicated thread. If it fills, rows are dropped rather than
back-pressuring inference: losing a log row is acceptable, dropping a frame is not. Only frame
summaries and track lifecycles are stored, never one row per box per frame, which at 30 FPS would
write millions of rows an hour for no analytical gain.

---

## 5. The frontend

Next.js 15 App Router, React 19, Tailwind v4 with CSS-variable semantic tokens.

**Theme.** Light and dark are tuned separately rather than one being an inversion of the other —
detection overlays have to stay legible in both. A blocking inline script sets `data-theme` before
first paint so there is no flash, and the control is three-state (light / system / dark) so choosing
a theme does not permanently strand you off your OS preference.

**Streaming render path.** Two decisions, both from measurement:

1. **Frames never enter React state.** Each annotated JPEG is painted straight to a canvas. Putting
   a ~100 KB data URI into state 30 times a second re-renders the tree every frame for no benefit.
2. **Painting is immediate; only telemetry is batched.** No `requestAnimationFrame` pacing —
   rAF is throttled or suspended in background and unfocused tabs, which stalls the visible stream
   exactly when someone is watching the network panel. Numeric readouts update at 4 Hz because no
   human reads a latency figure 30 times a second.

Async image decode means a slow frame can land after a newer one, so paints are gated on a monotonic
`frame_id` — dropping the late frame instead of flickering backwards.

---

## 6. Quantization: three things that were not what they appeared

The most valuable part of this project. Each was caught by inspecting artifacts rather than trusting
an export flag.

### 6.1 `half=True` is silently dropped on CPU export

The first FP16 and INT8 ONNX exports both came out at exactly 10.21 MB, which is the FP32 size.
Reading the graph confirmed it:

```python
Counter(TensorProto.DataType.Name(i.data_type) for i in model.graph.initializer)
# yolo11n_fp16.onnx -> {'FLOAT': 175}     # FP32 wearing an FP16 filename
```

Ultralytics drops `half=True` for ONNX unless the export runs on a CUDA device. With `device=0`, the
same check reports `{'FLOAT16': 175}` and the artifact halves to 5.14 MB.

### 6.2 `int8=True` is a no-op for ONNX

That flag applies to the TFLite and TensorRT paths. An "INT8 ONNX" produced that way is still FP32.

### 6.3 Dynamic quantization produces an unrunnable model

`quantize_dynamic` rewrites convolutions to `ConvInteger`, which ONNX Runtime's CPU provider has no
kernel for:

```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
```

Static quantization in **QDQ format** emits `QLinearConv`, which is supported.

### 6.4 The detection head must stay in float

The first working static-quantized model returned **zero** detections — it loaded and ran without
error, and recall against FP32 was 0.0% across 150 frames.

The box-regression branch of the detection head decodes distributions into coordinates, and 8-bit
activation ranges there collapse the geometry even though the backbone quantizes cleanly.

| | Recall | Precision | Mean IoU | Size |
|---|---|---|---|---|
| Head quantized | 0.0% | 0.0% | - | 3.07 MB |
| Head in float | **93.3%** | 88.8% | 0.956 | 4.36 MB |

The head is identified structurally, as the highest `/model.N/` index in the graph, so the exclusion
survives an architecture change.

### 6.5 Calibration hygiene

Calibration footage is deliberately **different** from evaluation footage. Calibrating on the clip
you then report accuracy against fits activation ranges to the exact frames under test and flatters
the result. Two clips are fetched: a dense 1080p scene for demo/evaluation, and OpenCV's `vtest.avi`
for calibration only. Preprocessing mirrors Ultralytics exactly (letterbox with pad 114, BGR->RGB,
0-1, CHW) — a mismatch collects statistics for inputs the model never receives. Frames are sampled
evenly across the clip, because the opening seconds of most footage are unrepresentative.

---

## 7. Measuring accuracy: why not cosine similarity

The original spec asked for output-tensor cosine similarity >= 0.99 as the export-parity check. That
metric is a poor decision signal for a detector: raw head outputs are dominated by thousands of
low-confidence anchors that never survive NMS, so a model can score 0.99 while losing the detections
that matter.

Replaced with agreement on **post-NMS detections**: greedy one-to-one matching per frame against the
FP32 baseline, same class, IoU >= 0.5.

- **Recall** — of the objects FP32 found, how many did this backend also find
- **Precision** — of this backend's detections, how many correspond to a real one
- **Mean IoU** — how tightly matched boxes agree

That is the question a deployment decision actually turns on.

---

## 8. Results

RTX A1000 Laptop GPU (4096 MiB), i9-12900H, 200 frames of 1080p footage, ~17 objects per frame, full
pipeline including decode and tracking, 10 warmup frames discarded.

| Backend | Size | Device | FPS | p95 ms | Peak VRAM | Recall | Precision | Mean IoU | Pareto |
|---|---|---|---|---|---|---|---|---|---|
| `fp32_gpu` | 640px | cuda | **48.5** | 21.2 | 316 MiB | 100.0% | 100.0% | 1.000 | yes |
| `fp32_gpu` | 480px | cuda | **71.7** | 14.9 | 306 MiB | 81.4% | 84.6% | 0.906 | yes |
| `fp32_gpu` | 320px | cuda | **84.3** | 13.1 | 312 MiB | 57.2% | 89.7% | 0.898 | yes |
| `openvino_cpu` | 640px | cpu | **35.8** | 31.1 | n/a | 98.4% | 92.4% | 0.984 | |
| `fp32_cpu` | 640px | cpu | **15.4** | 71.7 | n/a | 100.0% | 99.9% | 0.999 | |
| `int8_onnx_cpu` | 640px | cpu | **15.1** | 72.6 | n/a | 93.3% | 88.8% | 0.956 | |

### What the numbers actually say

**The 4 GB budget was never the constraint.** 316 MiB peak, about 9% of the budget. The design
assumed VRAM would be the binding limit; it is not. Compute and transport are.

**CPU inference clears real-time.** OpenVINO reaches 35.8 FPS at 98.4% recall with no GPU at all —
the opposite of the design assumption.

**Quantization did not pay off on this hardware.** INT8 is 2.4x slower than OpenVINO and gives up 5
points of recall for a 19% smaller artifact. On a memory-constrained edge target with no OpenVINO
support the trade flips, which is precisely why it is measured per-target rather than assumed.

**Resolution is the strongest lever and it is expensive.** 640->480 px buys 48% throughput for 19
points of recall; 640->320 px buys 74% for 43 points. On densely populated footage many objects are
small, so resolution cuts lose them outright.

---

## 9. Performance work on the streaming path

Initial viewer throughput was 8.8 FPS against a 48 FPS pipeline. Profiling each stage on 1080p:

| Stage | ms |
|---|---|
| Inference | 29.5 |
| Annotate | 8.1 |
| Downscale + JPEG + base64 | 27.7 |
| **Total** | **65.2 (15.3 FPS)** |

Two fixes:

1. **Stopped rebuilding the full metrics snapshot per frame.** The streaming loop called a method
   making several `psutil` syscalls, at the frame rate — more expensive than the inference it was
   reporting on. Replaced with a cheap rolling-FPS accessor.
2. **Moved the downscale to the capture thread.** The detector letterboxes to 640 px regardless, so
   inference, annotation and encoding were all paying for pixels the model never sees. Capping frame
   width at the producer measured **+39% viewer throughput**.

Result: server latency **101 ms -> 48 ms**, viewer **8.8 -> 13.4 FPS**, end-to-end send-to-paint
**~12 ms** against a 100 ms target.

**A measurement trap worth remembering:** an early profiling run reported 739 ms per frame instead of
30 ms, because it shared the 4 GB GPU with the running API. On a small GPU, contention is not noise
— it is the dominant term. Every benchmark here runs with the GPU otherwise idle.

---

## 10. Dependency archaeology

Four pinned versions in the original plan were wrong, and each failure was informative:

| Pin | Reality | Resolution |
|---|---|---|
| `hydra-core==1.3.5` | Does not exist; latest is 1.3.2 | Pinned 1.3.2 |
| `openvino 2026.0` | Removed `openvino.runtime`, which Ultralytics calls | Pinned 2024.6.0 |
| `onnxruntime-gpu 1.27` | Needs cuDNN 9; torch cu121 ships cuDNN 8.9 | Pinned 1.18.1 |
| `ultralytics==8.3.37` | Uses `EXPLICIT_BATCH`, removed in TensorRT 11 | Upgraded to 8.4.104 |

A separate hazard: **Ultralytics auto-installs dependencies at runtime.** Mid-session it ran
`pip install onnxruntime` over the GPU build and left an environment where even `import torch`
failed. `YOLO_AUTOINSTALL=false` is now set on package import. Dependency changes belong in
requirements files, never in a request handler.

---

## 11. Testing

**115 API tests plus 14 ML-conversion tests.** Split into a fast pure-logic tier (parsing, registry rules, decode, telemetry
bounds, capture semantics) and a slower tier that boots the real model through the FastAPI lifespan.
The slow tier proves the model loads, serves, hot-swaps and streams *on this machine* — including a
live WebSocket test that asserts frames arrive with real overlays and monotonic ids.

Tests assert on properties rather than fixtures where possible: that the fallback ladder ends on a
backend requiring neither GPU nor export step; that rolling buffers stay bounded under 3x their
window; that FPS is derived from wall-clock spacing rather than latency (a collector fed 1 ms
latencies 20 ms apart must report ~50 FPS, not 1000).

**5 Playwright browser tests** asserting behaviour, not markup: that the canvas receives pixels that
keep changing across samples, that the FPS readout leaves zero, that unavailable backends display a
reason, that the theme toggle restyles the document and survives reload, and that the page never
produces a double or horizontal scrollbar.

A test that only checks a heading exists would pass against a completely broken stream.

---

## 12. Honest gaps

- **TensorRT engines not built.** TRT 10 has no installable Windows wheel; TRT 11 removed the
  Ultralytics API and its replacement needs NVIDIA ModelOpt, which requires torch >= 2.8 and would
  replace the validated torch 2.3.1+cu121 stack. Registry, exporter and UI support it; artifacts are
  absent and reported as such.
- **FP16 ONNX on GPU does not load.** ONNX Runtime needs cuDNN 8 on `CUDA_PATH`; the installed CUDA
  12.0 toolkit does not ship it. Detected at warmup, reason shown in the UI.
- **Accuracy is agreement against an FP32 baseline on one clip**, not COCO mAP or MOT17 IDF1. Those
  need the evaluation datasets and belong to the training phase.
- **Stability verified over minutes, not the planned 4 hours.**
- **No fine-tuning.** The model is pretrained COCO.

---

## 13. Résumé bullets

- Built a real-time multi-object tracking system (YOLO11n + ByteTrack) achieving **48.5 FPS at 640 px
  using 316 MiB VRAM** on a 4 GB laptop GPU, with a browser console streaming annotated frames at
  ~12 ms end-to-end latency.
- Designed a **declarative backend registry with a fallback ladder** across TensorRT, ONNX, OpenVINO
  and PyTorch, where usability is proven by warmup rather than assumed by load, and failures are
  recorded per configuration and surfaced in the UI.
- Implemented **static QDQ INT8 quantization** with held-out calibration; diagnosed a total accuracy
  collapse (0% recall) to detection-head quantization and recovered **93.3% recall vs FP32** by
  excluding the head structurally.
- Caught that framework export flags silently produced FP32 graphs with quantized filenames, by
  **verifying artifact dtypes** rather than trusting the export API.
- Produced an **accuracy-throughput Pareto frontier** scored on post-NMS detection agreement instead
  of tensor cosine similarity, showing INT8 was the wrong choice on the target CPU and that an
  OpenVINO CPU path cleared real-time at 98.4% recall.
- Profiled and optimised the streaming path from **8.8 to 13.4 FPS** (server latency 101 ms -> 48 ms)
  by eliminating per-frame syscalls and moving downscaling ahead of inference.
- Shipped with **129 automated tests (115 API, 14 ML) and 5 browser E2E tests** asserting observable behaviour, plus
  documented degradation drills exercisable through the API.

## 14. Skills demonstrated

**Computer vision** — object detection, multi-object tracking, IoU matching, NMS behaviour,
letterbox preprocessing, detection-head architecture.
**Model optimization** — post-training static quantization, calibration design, QDQ vs dynamic
formats, per-channel weights, selective node exclusion, format export (ONNX, OpenVINO).
**Performance engineering** — stage-level profiling, GPU contention diagnosis, throughput/latency
trade-off analysis, Pareto frontier construction.
**Backend** — FastAPI, WebSocket streaming, threaded producer/consumer with bounded queues,
concurrency under a lock, async offloading of blocking work, graceful degradation, async persistence.
**Frontend** — Next.js App Router, React 19, canvas rendering pipelines, theme systems with CSS
variables, accessibility (contrast, focus, reduced motion, ARIA).
**Engineering practice** — dependency conflict resolution, artifact verification over API trust,
behaviour-driven testing, honest documentation of gaps.

---

Phu Nguyen - HCMC, Vietnam
