# MLOps: tracking, registry, and the promotion gate

Covers PRD **FR-16** (MLflow tracking + registry + promotion gate) and the **NFR-4** Hydra
requirement. Everything on this page runs **LOCAL** except where it says CLOUD.

| Step | Where | GPU | VRAM |
|---|---|---|---|
| Fine-tune (`ml/scripts/train_colab.py`) | **CLOUD** (Colab T4) | required | ~10-14 GB |
| Quantize + export (`ml/quantization/`) | LOCAL-4GB | optional | < 1.5 GB |
| Evaluate (`ml/eval/eval_coco.py`, `eval_mot.py`) | LOCAL-4GB | optional | < 1.5 GB |
| Tracking server + registry | LOCAL | none | none |
| Promotion gate (`benchmark_precision.py`) | LOCAL | none | none |

---

## 1. Why a database-backed tracking server

MLflow's **Model Registry is not implemented on the bare `mlruns/` file store**. `register_model`
and `transition_model_version_stage` both require a database backend. That is the whole reason this
project runs a server instead of pointing at a directory, and it is why **model registration cannot
happen on Colab** — a Colab session has no route to your laptop's `127.0.0.1:5000`.

### Start it

```powershell
# From the repo root. Print the exact command instead of copying it by hand:
.\.venv\Scripts\python.exe ml\quantization\benchmark_precision.py --print-server-command

# Or let the gate run it in the foreground (Ctrl-C to stop):
.\.venv\Scripts\python.exe ml\quantization\benchmark_precision.py --start-server
```

The command it prints and runs is:

```powershell
.\.venv\Scripts\python.exe -m mlflow server `
  --backend-store-uri sqlite:///D:/FPT/Demo/streamsight/mlflow.db `
  --artifacts-destination file:///D:/FPT/Demo/streamsight/mlartifacts `
  --serve-artifacts --host 127.0.0.1 --port 5000
```

Then point clients at it:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

The UI is at <http://127.0.0.1:5000>.

### Two Windows details that cost real debugging time

**Artifacts are proxied, not addressed by path.** The PRD's example line uses
`--default-artifact-root ./mlartifacts`. With an *absolute* Windows path there, MLflow stores the
experiment's artifact location back as `d:/FPT/Demo/streamsight/mlartifacts/...`, then resolves
artifact repositories by URI scheme and reads `d` as the scheme. Every `log_artifact` fails with
`Could not find a registered artifact repository for: d:/...` — and it fails *after* the run row has
already been created, so you are left with empty runs. Observed on mlflow 2.17.0. A relative root
avoids it only while every client shares the server's working directory. `--serve-artifacts` with
`--artifacts-destination` makes the client-side URI `mlflow-artifacts:/`, which has no drive letter,
so this class of failure disappears.

**An experiment's artifact location is fixed at creation.** If you started the server with a broken
artifact root once, changing the flag is not enough — the stored location persists in `mlflow.db`.
Stop the server, delete `mlflow.db` and `mlartifacts/`, and start again.

**`mlflow server` leaves worker processes behind.** It runs under `waitress` on Windows; killing the
parent leaves the workers holding port 5000 and a lock on `mlflow.db`. To stop it completely:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*waitress*mlflow*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## 2. Registry model names

| Registry model | What it holds | Gated on | Blocks the API? |
|---|---|---|---|
| `streamsight-detector` | the exported detector artifact (`yolo11n_int8.onnx`, or the FP16 sibling) | COCO mAP50-95 drop vs the local FP32 baseline | **no** — see below |
| `streamsight-tracker-quality` | the ByteTrack config (`ml/models/config/bytetrack.yaml`) | MOT17 IDF1 | **no** — advisory only |

> **The API does not read the registry.** `apps/api/app/backends.py` resolves artifacts from fixed
> paths on disk; there is no MLflow code anywhere under `apps/api`. Promoting a version records a
> decision, it does not change what serves. To deploy a promoted artifact an operator copies it into
> `ml/models/engines/` and restarts, or hot-swaps via `POST /config/model`.
>
> A registry-aware loader is a reasonable next step and is deliberately not implemented: it would
> couple request-path startup to an MLflow server being reachable, which is the wrong trade for a
> single-node local service. PRD FR-16's closing clause ("and the API loads that engine") is
> therefore **not met** — recorded here rather than papered over.

Tracking quality is a *separate* model on purpose. IDF1 measures association across frames, which
quantization does not control; letting a weak IDF1 archive the detector would take inference offline
for a defect that swapping engines cannot fix. This split is PRD FR-16, not an invention here.

> `transition_model_version_stage` is deprecated since MLflow 2.9 in favour of aliases, and emits a
> `FutureWarning`. It is used anyway because the PRD names stages explicitly. If stages are removed
> in MLflow 3.x, the replacement is `set_registered_model_alias(name, "production", version)`.

---

## 3. The promotion condition

Read verbatim from `ml/train/config.yaml` (`gate.map_drop_max`, default `0.03`):

> The INT8 candidate is transitioned to **Production** if and only if
> **`fp32_map50_95 - int8_map50_95 <= 0.03` in ABSOLUTE terms**, with both numbers measured
> **locally** by `ml/eval/eval_coco.py` **on the same class set**.
>
> If that fails, the **FP16** sibling is transitioned to Production instead, so there is always a
> Production version for the API to load.

Three things this deliberately is *not*:

- **Not relative.** A 3% *relative* drop from 0.412 would be 0.0124, four times stricter. The PRD
  says absolute.
- **Not against a published number.** The deployed detector is a 6-class fine-tune. The 80-class
  YOLO11n reference (39.5% @640) is a sanity check, never the baseline. The gate refuses to compare
  two reports whose `class_set` or `classes` differ, and says so:
  `class-set mismatch: baseline 'fp32_gpu' ... was scored on [...] but candidate ... on [...]`.
- **Not gated on tracking.** MOT17 IDF1 (`gate.mot_idf1_min`, default `0.60`) is logged and
  registered, and its verdict is printed as `advisory only`.

### Run it

```powershell
# 1. Produce the eval reports (one JSON per backend) — see docs/BENCHMARKS.md.
.\.venv\Scripts\python.exe ml\eval\eval_coco.py --engine fp32_gpu      --classes prd6
.\.venv\Scripts\python.exe ml\eval\eval_coco.py --engine int8_onnx_cpu --classes prd6
.\.venv\Scripts\python.exe ml\eval\eval_coco.py --engine fp16_onnx     --classes prd6
.\.venv\Scripts\python.exe ml\eval\eval_mot.py  --engine fp32_gpu

# 2. Decide without touching MLflow. Works with no server running.
.\.venv\Scripts\python.exe ml\quantization\benchmark_precision.py --dry-run

# 3. Register + transition for real.
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
.\.venv\Scripts\python.exe ml\quantization\benchmark_precision.py
```

Flags: `--candidate` (default `int8_onnx_cpu`), `--baseline` (default `fp32_gpu`), `--fallback`
(default `fp16_onnx`), `--coco-report` / `--mot-report` (file or directory, repeatable;
default `ml/eval/reports/`), `--skip-tracker`, `--out` (default
`ml/quantization/reports/promotion.json`).

Sample output of the failing branch:

```text
baseline  fp32_gpu         mAP50-95 0.4120
candidate int8_onnx_cpu    mAP50-95 0.3300
drop      8.20pp (limit 3.00pp)
verdict   FAIL - INT8 mAP50-95 drop 8.20pp > 3.00pp; falling back to FP16
promote   fp16_onnx
```

> The numbers above come from a synthetic fixture used to exercise both branches of the gate. They
> are **not** measured accuracy. Real figures live in the JSON that `eval_coco.py` writes.

### What the gate reads

`eval_coco.py` writes one report per backend, `ml/eval/reports/coco_<backend>_<imgsz>.json`, with a
flat record carrying `engine`, `map50_95`, `map50`, `imgsz`, `class_set`, and `classes`. The gate
globs that directory and merges the records, newest file winning per backend.

The reader is intentionally unforgiving. A missing key would promote an engine on a default value,
so anything unexpected aborts with exit code 2 and a message naming the file, the keys it looked
for, and the keys it found:

```text
gate contract violated: <path>: a record is missing required fields. Need a backend identifier
(one of ['engine', 'backend', 'backend_key', 'name']) and mAP50-95 (one of ['map50_95', ...]).
Record keys: ['accuracy', 'engine']
```

Metrics given as percentages (`41.2`) are normalised to fractions (`0.412`); values above 1 can only
be percentages, and reading `41.2` as a fraction would let every candidate through.

### What it writes

`ml/quantization/reports/promotion.json` — the decision, every number behind it, the reports it
read, and (outside `--dry-run`) the registered versions. Written in dry-run mode too, so the verdict
is auditable without a server.

The gate does **not** copy artifacts: the exported engines already live in `ml/models/engines/`, and
copying a file onto itself would only add a way for the two to diverge. The promoted backend key is
recorded in the report and in the run's `backend` param.

---

## 4. What Colab does vs what runs locally

**CLOUD (Colab, `ml/scripts/train_colab.py`):**

- fine-tunes pretrained `yolo11n.pt` on the 6-class person+vehicle subset;
- logs params and per-epoch metrics to its **own ephemeral** MLflow store;
- checkpoints every `train.checkpoint_every` epochs so a reclaimed session costs one interval;
- produces **`best.pt`**, which you download manually into `ml/models/weights/`.

It registers **nothing**. There is no registry on Colab to register into.

**LOCAL:** quantize → export → evaluate → `benchmark_precision.py` registers and transitions.

Resume after a reclaimed session:

```powershell
python ml/train/trainer.py train.resume=true
```

Resuming with no checkpoint present starts fresh rather than erroring, so the notebook can run the
same command every session.

---

## 5. Hydra configuration (NFR-4)

`ml/train/config.yaml` is the single Hydra config. `trainer.py` consumes it through `@hydra.main`;
`benchmark_precision.py` reads the same file through OmegaConf for `gate.*` and `mlflow.*`. One file
rather than three because the accepted accuracy drop must mean the same thing to the trainer and to
the gate.

```powershell
python ml/train/trainer.py --dry-run                      # no GPU, no dataset needed
python ml/train/trainer.py train.epochs=20 train.batch=8
python ml/train/trainer.py model.imgsz=480 seed=7
```

`--dry-run` is translated to `dry_run=true` before Hydra parses argv, so the flag matches every
other script in this repo. Hydra writes its run record to `ml/train/outputs/<timestamp>/`.

Sections: `seed`, `model`, `data` (including the 6-class list), `train`, `gate`, `mlflow`.

Hydra is applied **here only**. Wrapping the export and benchmark scripts in it as well would add
config plumbing to tools that take three arguments each — one genuine use is worth more than three
ceremonial ones.

---

## 6. Files

| Path | Role |
|---|---|
| `ml/train/config.yaml` | Hydra config: training hyperparameters **and** gate thresholds |
| `ml/scripts/train_colab.py` | the Colab deliverable: builds the 6-class subset on the VM, checkpoints to Drive every 5 epochs, resumes by default. Never executed. |
| `ml/train/trainer.py` | LOCAL Hydra-configured entry point, used for `--dry-run` config validation and local experiments. Colab does NOT call it. |
| `ml/quantization/benchmark_precision.py` | the promotion gate; registry server helper |
| `ml/eval/eval_coco.py` | COCO mAP per backend — the gate's accuracy source |
| `ml/eval/eval_mot.py` | MOT17 MOTA/IDF1/IDSW — the tracker gate's source |
| `mlflow.db`, `mlartifacts/` | tracking + registry state (git-ignored) |

Phu Nguyen - HCMC, Vietnam

