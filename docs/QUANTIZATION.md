# Quantization and export

Every claim here was verified by inspecting the produced artifacts, not by trusting an export flag.
Two of the findings below only surfaced because of that.

## Exporting

```powershell
python ml/quantization/export_engines.py --formats fp16_onnx openvino_cpu
python ml/quantization/calibrate.py --frames 128
```

| Target | Route | Artifact | Size |
|---|---|---|---|
| `fp16_onnx` | Ultralytics export, `half=True`, `device=0` | `yolo11n_fp16.onnx` | 5.14 MB |
| `openvino_cpu` | Ultralytics export, OpenVINO IR | `yolo11n_openvino_model/` | 5.45 MB |
| `int8_onnx_cpu` | ONNX Runtime static QDQ, calibrated | `yolo11n_int8.onnx` | 4.36 MB |
| `fp16_trt` / `int8_trt` | Ultralytics engine export | not built here, see below |

FP32 ONNX for reference is 10.2 MB, so the FP16 artifact at 5.14 MB is genuinely half-precision.

## Three things that were not what they appeared

### 1. `half=True` is silently ignored when exporting ONNX from the CPU

The first FP16 and INT8 exports both came out at exactly 10.21 MB, which is the FP32 size. Reading
the graph initializers confirmed it:

```python
Counter(TensorProto.DataType.Name(i.data_type) for i in model.graph.initializer)
# yolo11n_fp16.onnx -> {'FLOAT': 175}     <- FP32 wearing an FP16 filename
```

Ultralytics drops `half=True` for the ONNX format unless the export runs on a CUDA device. Passing
`device=0` fixes it, and the same check now reports `{'FLOAT16': 175}`.

### 2. Ultralytics' `int8=True` does nothing for ONNX

That flag applies to the TFLite and TensorRT paths. An "INT8 ONNX" produced that way is still FP32.
Real 8-bit requires ONNX Runtime.

### 3. Dynamic quantization produces a model that cannot run

The obvious next move, `quantize_dynamic`, rewrites convolutions to `ConvInteger`, for which ONNX
Runtime's CPU provider has no kernel:

```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
```

Static quantization in **QDQ format** is the correct route for a CNN: it emits `QLinearConv`, which
is supported and genuinely faster.

## Calibration

Static quantization needs representative activation ranges, so it needs data.

**Calibration footage is deliberately different from evaluation footage.** Calibrating on the same
clip used to report accuracy fits the activation ranges to the exact frames under test and flatters
the result. `fetch_assets.py` therefore downloads two clips: a dense 1080p classroom scene for demo
and evaluation, and OpenCV's `vtest.avi` pedestrian plaza for calibration only.

Preprocessing mirrors Ultralytics exactly (letterbox to a square with pad value 114, BGR to RGB,
scale to 0-1, CHW). A mismatch would collect statistics for inputs the model never receives. Frames
are sampled evenly across the clip rather than taken from the opening seconds, which in most footage
are unrepresentative.

## The detection head stays in float

The first working static-quantized model returned **zero** detections. Recall against the FP32
baseline was 0.0% across 150 frames while the model loaded and ran without error.

The cause is the detection head. Its box-regression branch decodes distributions into coordinates,
and 8-bit activation ranges there collapse the geometry even though the backbone quantizes cleanly.
Excluding the final module's 108 nodes from quantization recovered it:

| | Recall vs FP32 | Precision | Mean IoU | Size |
|---|---|---|---|---|
| Head quantized | 0.0% | 0.0% | - | 3.07 MB |
| Head kept in float | **93.3%** | 88.8% | 0.956 | 4.36 MB |

The head is identified structurally, as the highest `/model.N/` index in the graph, so the exclusion
keeps working if the architecture depth changes.

## What INT8 actually cost in accuracy

Standard COCO mAP on the PRD's six-class subset (2,968 val2017 images, `conf=0.001`,
`max_det=100`), which is the number the promotion gate reads:

| Backend | mAP50-95 | mAP50 | Absolute drop | Artifact |
|---|---|---|---|---|
| `fp32_gpu` | 0.4211 | 0.6105 | baseline | 5.35 MB |
| `openvino_cpu` | 0.4211 | 0.6133 | 0.00 pp | 5.45 MB |
| `int8_onnx_cpu` | **0.4172** | 0.6080 | **0.40 pp** | 4.36 MB |

**0.40 points against a 3.00-point gate.** Keeping the detection head in float and calibrating on
held-out footage did their job: 8-bit weights cost essentially nothing here. OpenVINO's FP16 IR is
lossless to four decimals.

Note this is a *different* measurement from the recall figures below. mAP integrates the whole
precision/recall curve at `conf=0.001`; recall-vs-FP32 is agreement on the boxes the app actually
serves at its 0.25 threshold. INT8 loses 0.40 pp of mAP but 6.7 pp of served-box agreement, because
quantization perturbs scores most where they are near the serving threshold. Both are true; they
answer different questions.

## Was INT8 worth it here?

On accuracy, yes: 0.40 pp is a rounding error. On **throughput**, measured on this machine: **no**,
and the frontier is what makes that answerable.

| Backend | FPS | Recall | Artifact |
|---|---|---|---|
| OpenVINO CPU FP16 | 35.8 | 98.4% | 5.45 MB |
| INT8 ONNX CPU | 15.1 | 93.3% | 4.36 MB |
| FP32 PyTorch CPU | 15.4 | 100.0% | 5.35 MB |

INT8 buys a 19% smaller artifact and loses more than half the throughput of OpenVINO, while giving
up 5 points of recall. On an i9-12900H, OpenVINO's optimized kernels beat ONNX Runtime's quantized
ones. On a memory-constrained edge target with no OpenVINO support the trade flips, which is exactly
why the answer is measured per-target rather than assumed.

## Per-target fit

Which export to reach for, and what each one actually cost here. Every number in this table comes
from `ml/eval/reports/frontier.json` (200 frames of 1080p footage, full pipeline including decode,
letterbox, inference and ByteTrack association) or from the artifact sizes above. Cells with no
number are cells where nothing was measured - they say so rather than carrying an estimate.

Throughput and recall are at **640 px**. Recall is against the `fp32_gpu@640` baseline, matched
greedily one-to-one on class and IoU >= 0.5, as described in `docs/BENCHMARKS.md`. Sizes are MiB.
The host is an RTX A1000 Laptop GPU (4096 MiB) with an Intel i9-12900H.

| Export | Target hardware | Artifact | Throughput | Recall vs FP32 | Choose it when | On this host |
|---|---|---|---|---|---|---|
| **ONNX FP16** | NVIDIA GPU through ONNX Runtime's CUDA provider; the graph itself is portable to any ORT-supported runtime | 5.14 MB | not measured | not measured | You need a portable graph and a GPU runtime that is not TensorRT, or half the FP32 file size for the same architecture | **Built, does not run.** The frontier sweep failed to create the CUDA provider: `CUDA_PATH is set but CUDA wasnt able to be loaded`. Not diagnosed further - `fp32_gpu` covers the GPU path and is faster to reach |
| **ONNX INT8** | x86 CPU through ORT's CPU provider (`QLinearConv`); the smallest artifact, so also the memory-constrained target | 4.36 MB | 15.1 FPS | 93.3% (mAP50-95 0.4172) | The target has no OpenVINO support and artifact size or memory is the binding constraint | **Built and running.** In the backend ladder as `int8_onnx_cpu`; the Production version in the MLflow registry |
| **OpenVINO** | Intel CPU (and Intel iGPU/NPU through the same IR) | 5.45 MB | 35.8 FPS | 98.4% (mAP50-95 0.4211) | The target is an Intel CPU. On this machine it is the best CPU option by a wide margin - 2.4x the INT8 throughput with 5 points more recall | **Built and running.** In the ladder as `openvino_cpu`; reaches 35.8 FPS with no GPU at all |
| **TensorRT** | NVIDIA GPU with CUDA 12.x | not built on this host | not built on this host | not built on this host | The target is an NVIDIA GPU on a platform where TensorRT 10 installs - it is the fastest NVIDIA path and the PRD's intended INT8 route | **Not built.** TensorRT 10 has no installable Windows wheel; TensorRT 11 removes an API Ultralytics 8.3.37 calls, and the 11 path needs ModelOpt, which needs torch >= 2.8 against a pinned torch 2.3.1 stack. Full reasoning below |

For reference on the same run, the unquantized baselines: `fp32_gpu` 48.5 FPS at 5.35 MB, `fp32_cpu`
15.4 FPS at 5.35 MB. The honest summary of this table on *this* hardware is that quantization did
not win: FP32 on the GPU is the fastest path, OpenVINO is the best CPU path, and INT8 only pays off
on a target that has neither - which is precisely why the matrix reports per-target rather than
declaring a winner.

## TensorRT

Not built on this machine, and the reason is a genuine toolchain dead end rather than an oversight:

- **TensorRT 10** has no installable Windows wheel on PyPI or NVIDIA's index. Every 10.x version
  resolves to a source distribution whose build fails.
- **TensorRT 11** installs cleanly but removed `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`, which
  Ultralytics 8.3.37 calls. Upgrading to Ultralytics 8.4.104 clears that, but the TRT 11 path then
  requires **NVIDIA ModelOpt**, which depends on **torch >= 2.8** and would replace the pinned
  torch 2.3.1 + cu121 stack the rest of the project is validated against.

The backend registry, the export script and the model-selector UI all support TensorRT. The
artifacts are simply absent, and every surface reports them as unavailable with that reason rather
than pretending otherwise.

Phu Nguyen - HCMC, Vietnam
