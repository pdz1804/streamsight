# StreamSight task runner.
#
# make is a convenience, not a dependency. Windows is the primary shell here and
# does not ship make, so every target below is a thin wrapper around a command
# that is also spelled out in README.md and docs/DEPLOYMENT.md -- run those
# directly if you would rather not install make.
#
# The interpreter is resolved from the project venv so the targets behave the
# same whether the venv is activated or not.

PY := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)

# Overridable on the command line, e.g. `make bench ENGINE=openvino_cpu FRAMES=300`.
ENGINE  ?= fp32_gpu
IMGSZ   ?= 640
FRAMES  ?= 200
SOAK    ?= 14400
FORMATS ?= fp16_onnx openvino_cpu

.PHONY: help setup api web test test-fast lint fmt bench frontier soak export

help:
	@echo setup - create .venv, install deps, fetch weights and clips
	@echo api - run the FastAPI server on :8100
	@echo web - run the Next.js console on :3100
	@echo test - full pytest suite, needs model weights
	@echo test-fast - pytest without the model-backed tier, what CI runs
	@echo lint - ruff + black --check
	@echo fmt - black
	@echo bench - single-backend throughput benchmark
	@echo frontier - accuracy/throughput sweep across backends
	@echo soak - long-running stability probe against a live API
	@echo export - export the CPU backends and calibrate INT8

# The torch index is the CUDA 12.1 line, matching the driver this project was
# built against. Swap /cu121 for /cpu on a machine without an NVIDIA GPU.
setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch==2.3.1 torchvision==0.18.1 \
		--index-url https://download.pytorch.org/whl/cu121
	$(PY) -m pip install -r requirements.txt
	$(PY) ml/scripts/fetch_assets.py
	cd apps/web && npm install

api:
	$(PY) -m uvicorn app.main:app --app-dir apps/api --port 8100

web:
	cd apps/web && npm run dev

test:
	$(PY) -m pytest apps/api/tests -q

test-fast:
	$(PY) -m pytest apps/api/tests -q -m "not slow"

lint:
	$(PY) -m ruff check apps/api ml
	$(PY) -m black --check apps/api ml

fmt:
	$(PY) -m black apps/api ml

bench:
	$(PY) ml/scripts/benchmark_inference.py --engine $(ENGINE) --imgsz $(IMGSZ) --frames $(FRAMES)

frontier:
	$(PY) ml/eval/benchmark_frontier.py --frames $(FRAMES)

# Requires the API to already be serving on :8100 (`make api` in another shell).
soak:
	$(PY) ml/scripts/soak_stream.py --duration $(SOAK)

export:
	$(PY) ml/quantization/export_engines.py --formats $(FORMATS)
	$(PY) ml/quantization/calibrate.py
