# Benchmarks

## How to reproduce

```powershell
# One configuration
python ml/scripts/benchmark_inference.py --engine fp32_gpu --imgsz 640 --frames 300

# Full sweep, agreement scoring, Pareto front, plot
python ml/eval/benchmark_frontier.py --frames 200 --imgsz 640 480 320
```

Outputs land in `ml/eval/reports/` as `frontier.md`, `frontier.json` and `frontier.png`.

## Test conditions

- NVIDIA RTX A1000 Laptop GPU, 4096 MiB, driver 573.44, CUDA 12
- Intel i9-12900H
- 200 frames of 1080p footage, ~17 objects per frame
- **Full pipeline**: decode, letterbox, inference, ByteTrack association. Not detector-only.
- 10 warmup frames discarded so first-call allocation and kernel selection do not skew short runs.

Nothing else was using the GPU. This matters more than it sounds: an early profiling run that shared
the 4 GB card with the running API reported 739 ms per frame instead of 30 ms. On a small GPU,
contention is not noise, it is the dominant term.

## Results

| Backend | Size | Device | FPS | p95 ms | Peak VRAM | Recall | Precision | Mean IoU | Pareto |
|---|---|---|---|---|---|---|---|---|---|
| `fp32_gpu` | 640px | cuda | **48.5** | 21.2 | 316 MiB | 100.0% | 100.0% | 1.000 | yes |
| `fp32_gpu` | 480px | cuda | **71.7** | 14.9 | 306 MiB | 81.4% | 84.6% | 0.906 | yes |
| `fp32_gpu` | 320px | cuda | **84.3** | 13.1 | 312 MiB | 57.2% | 89.7% | 0.898 | yes |
| `openvino_cpu` | 640px | cpu | **35.8** | 31.1 | n/a | 98.4% | 92.4% | 0.984 | |
| `fp32_cpu` | 640px | cpu | **15.4** | 71.7 | n/a | 100.0% | 99.9% | 0.999 | |
| `int8_onnx_cpu` | 640px | cpu | **15.1** | 72.6 | n/a | 93.3% | 88.8% | 0.956 | |
| `fp32_cpu` | 480px | cpu | **20.9** | 53.4 | n/a | 81.4% | 84.6% | 0.906 | |
| `fp32_cpu` | 320px | cpu | **27.0** | 42.7 | n/a | 57.2% | 89.7% | 0.899 | |

## Why agreement, not cosine similarity

The original plan asked for output-tensor cosine similarity >= 0.99 as the export-parity check. That
metric is a poor decision signal for a detector: raw head outputs are dominated by the thousands of
low-confidence anchors that never survive NMS, so a model can score 0.99 while losing the detections
that actually matter.

This measures agreement on **post-NMS detections** instead. For each frame, greedy one-to-one
matching against the FP32 baseline with the same class and IoU >= 0.5:

- **Recall** - of the objects FP32 found, how many did this backend also find
- **Precision** - of this backend's detections, how many correspond to a real baseline detection
- **Mean IoU** - how tightly the matched boxes agree

That is the question a deployment decision turns on.

## What the numbers say

**The 4 GB budget is not the constraint.** Peak VRAM is 316 MiB against a 3.5 GB budget, about 9%.
The model is 2.6 M parameters; the memory headroom was never at risk, and the interesting limits are
compute and transport instead.

**Resolution is the strongest throughput lever, and it is expensive.** 640 -> 480 px buys 48% more
throughput for 19 points of recall; 640 -> 320 px buys 74% for 43 points. On this densely populated
footage many objects are small, so dropping resolution loses them entirely. The `fp32_gpu` points
form the whole Pareto front because they are the only configurations free to vary resolution at all.

**CPU inference clears real-time.** OpenVINO reaches 35.8 FPS at 98.4% recall without touching the
GPU. The design assumed a GPU was required for 30 FPS; on this hardware that assumption was wrong.

**Quantization did not pay off here.** See [QUANTIZATION.md](QUANTIZATION.md): INT8 is 2.4x slower
than OpenVINO and gives up 5 points of recall for a 19% smaller artifact.

## Viewer throughput is a different number

The table above measures the detect-and-track pipeline. The browser viewer is slower because
annotation, JPEG encoding and WebSocket transport are on its critical path.

Measured per stage on 1080p source, before optimisation:

| Stage | ms |
|---|---|
| Inference | 29.5 |
| Annotate | 8.1 |
| Downscale + JPEG + base64 | 27.7 |
| **Total** | **65.2 (15.3 FPS)** |

Two changes were made off the back of that profile:

1. **Stopped rebuilding the full metrics snapshot per frame.** The streaming loop was calling a
   method that makes several psutil syscalls, at the frame rate. It now reads a cheap rolling FPS.
2. **Moved the downscale to the capture thread.** The detector letterboxes to 640 px regardless, so
   inference, annotation and encoding were all paying for pixels the model never sees.

Result: server latency **101 ms -> 48 ms**, viewer throughput **8.8 -> 13.4 FPS**, end-to-end
send-to-paint latency **~12 ms**, comfortably inside the 100 ms target.

### A third change, not yet re-measured

The transport was then rebuilt again: base64 data URIs replaced by one binary message per frame
(length-prefixed JSON header + raw JPEG), and the streaming pump split into a producer task
(capture + inference + annotation) and a consumer task (encode + send) so the two overlap instead of
running in series. See [ARCHITECTURE](ARCHITECTURE.md).

**The 13.4 FPS figure above predates that change and has not been superseded by a measurement.** The
attempt to re-measure was invalid and is being repeated: the host's discrete GPU was stuck near its
210 MHz floor against a 2100 MHz ceiling — cool at 52 °C, so power-starved rather than thermally
throttled, with the battery at 14% on a charging adapter. In that state the *unchanged* pipeline
benchmark read **8.07 FPS against its own 48.5 FPS baseline**, so every throughput number taken then
describes the host, not the code.

Bytes on the wire were measured in the same session and are not clock-sensitive: the binary
transport carries **0.76x the bytes per frame** of base64, matching the expected 1/1.33 overhead.
Re-measure with `python ml/scripts/measure_stream_delivery.py` (delivery rate) and the Playwright
`viewer paint rate` check (what a browser actually paints); confirm `nvidia-smi` shows the GPU
boosting first.

## Measurement caveats

- **Peak VRAM is host-wide, not process-scoped.** It comes from NVML
  (`nvmlDeviceGetMemoryInfo().used`), which reports every allocation on the device. If anything
  else is on the GPU during a run, the figure is inflated. Treat the numbers above as an upper
  bound measured on an otherwise-idle card.
- **Recall is measured on the output the app actually serves**, which is the post-NMS detection
  list returned by `model.track(...)`. That call applies the tracker, so a detection the tracker
  discards does not appear. This is deliberate: it scores what a user of this service receives,
  not what the raw detector head produced.
- **One clip, one scene.** Agreement is measured against an FP32 baseline on the bundled demo
  footage. It says which backend best reproduces the reference *on this content*; it is not a
  dataset-level accuracy claim.

## COCO mAP

Standard, comparable accuracy on the PRD's six-class person+vehicle subset. Measured with
pycocotools over the **2,968 val2017 images** that contain at least one of those classes
(14,323 annotations), at the COCO protocol's `conf=0.001`, `max_det=100`.

```powershell
python ml/data/scripts/download_coco.py
python ml/data/scripts/prepare_coco_subset.py
python ml/eval/eval_coco.py --engine fp32_gpu --classes prd6 \
  --annotations ml/data/processed/coco_person_vehicle/instances_val2017_person_vehicle.json
```

| Backend | mAP50-95 | mAP50 | Absolute drop vs FP32 | Artifact |
|---|---|---|---|---|
| `fp32_gpu` (baseline) | **0.4211** | 0.6105 | — | 5.35 MB |
| `openvino_cpu` | **0.4211** | 0.6133 | 0.00 pp | 5.45 MB |
| `int8_onnx_cpu` | **0.4172** | 0.6080 | **0.40 pp** | 4.36 MB |

Reports: `ml/eval/reports/coco_<backend>_640.json`.

**INT8 costs 0.40 percentage points of mAP**, against a 3.00 pp gate. That is the number the
promotion gate turns on, and it is what makes the quantization work defensible: the detection head
stays in float and calibration uses held-out footage, and the result is a 19% smaller artifact for
a fractional accuracy cost. OpenVINO's FP16 IR is lossless to four decimal places.

For orientation, YOLO11n's published **80-class** figure is 39.5% mAP50-95. This 42.1% is a
**different, easier problem** (six well-represented classes) and the two are not comparable — the
PRD is explicit that baseline and target must share a class set, and the evaluated set is recorded
in every report JSON so the gate can refuse a mismatched comparison.

## Honest gaps

- No TensorRT numbers; the artifacts could not be built on this host (see QUANTIZATION.md).
- No FP16 GPU numbers; ONNX Runtime cannot load a CUDA provider against the installed cuDNN.
- **No MOT17 MOTA/IDF1.** `ml/eval/eval_mot.py` is written and unit-tested, but MOTChallenge is
  registration-gated: the dataset cannot be fetched unattended. Supply the zip to
  `ml/data/scripts/download_mot.py --zip` and the numbers follow.
- The mAP above is on **val2017**, i.e. the pretrained COCO model on COCO data. It is a valid
  quantization comparison; it is not evidence about a fine-tuned model, which does not exist yet.

Phu Nguyen - HCMC, Vietnam
