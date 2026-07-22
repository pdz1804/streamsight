# Accuracy-throughput frontier

Generated 2026-07-22 05:04 UTC on NVIDIA RTX A1000 Laptop GPU.

- Frames per configuration: 200
- Source clip: `sample.mp4`
- Baseline: `fp32_gpu@640`
- Match rule: same class and IoU >= 0.5

| Backend | Size | Device | Artifact | FPS | p95 ms | Peak VRAM | Recall | Precision | Mean IoU | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|
| `fp32_gpu` | 640px | cuda | 5.35 MB | **48.5** | 21.17 | 316 MiB | 100.0% | 100.0% | 1.000 | yes |
| `int8_trt` | 640px | cuda | - | unavailable | - | - | - | - | - | - |
| `int8_trt` | 480px | cuda | - | unavailable | - | - | - | - | - | - |
| `int8_trt` | 320px | cuda | - | unavailable | - | - | - | - | - | - |
| `fp16_trt` | 640px | cuda | - | unavailable | - | - | - | - | - | - |
| `fp16_trt` | 480px | cuda | - | unavailable | - | - | - | - | - | - |
| `fp16_trt` | 320px | cuda | - | unavailable | - | - | - | - | - | - |
| `fp16_onnx` | 640px | cuda | - | unavailable | - | - | - | - | - | - |
| `fp32_gpu` | 480px | cuda | 5.35 MB | **71.66** | 14.93 | 306 MiB | 81.4% | 84.6% | 0.906 | yes |
| `fp32_gpu` | 320px | cuda | 5.35 MB | **84.25** | 13.08 | 312 MiB | 57.2% | 89.7% | 0.898 | yes |
| `openvino_cpu` | 640px | cpu | 5.45 MB | **35.79** | 31.14 | 308 MiB | 98.4% | 92.4% | 0.984 |  |
| `int8_onnx_cpu` | 640px | cpu | 4.36 MB | **15.09** | 72.61 | 308 MiB | 93.3% | 88.8% | 0.956 |  |
| `fp32_cpu` | 640px | cpu | 5.35 MB | **15.42** | 71.65 | 308 MiB | 100.0% | 99.9% | 0.999 |  |
| `fp32_cpu` | 480px | cpu | 5.35 MB | **20.85** | 53.43 | 260 MiB | 81.4% | 84.6% | 0.906 |  |
| `fp32_cpu` | 320px | cpu | 5.35 MB | **27.04** | 42.71 | 260 MiB | 57.2% | 89.7% | 0.899 |  |

## Configurations that could not run

- `int8_trt`: artifact missing (run ml/quantization exports)
- `fp16_trt`: artifact missing (run ml/quantization exports)
- `fp16_onnx`: D:\a\_work\1\s\onnxruntime\python\onnxruntime_pybind_state.cc:891 onnxruntime::python::CreateExecutionProviderInstance CUDA_PATH is set but CUDA wasnt able to be loaded. Please install the correct version of CUDA andcuDN

Phu Nguyen - HCMC, Vietnam
