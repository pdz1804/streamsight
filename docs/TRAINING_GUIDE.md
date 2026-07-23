# Training guide

Fine-tuning the detector on the COCO person+vehicle 6-class subset, and getting the result back into
the local pipeline.

**Training never runs on this machine.** The 4 GB laptop GPU serves inference; it does not have the
memory or the hours for a training run, which is a stated non-goal rather than a limitation to work
around. The script is `ml/scripts/train_colab.py` and it runs on a free Colab T4.

**Nothing in this guide is a measurement yet.** The script has not been executed - it is written to
be run by you, on your Google account. Every duration below is a budget carried over from the PRD,
labelled as such. Replace them with your observed numbers after the first run, and say where they
came from.

## CLOUD vs LOCAL, per step

| # | Step | Where | Runs on |
|---|---|---|---|
| 1 | Build the 6-class COCO subset | **CLOUD** | Colab VM disk, `/content` |
| 2 | Fine-tune `yolo11n.pt`, 10 epochs, batch 4 | **CLOUD** | Colab T4, 16 GB |
| 3 | Checkpoint every 5 epochs | **CLOUD** | written to Google Drive |
| 4 | Log params + metrics to MLflow | **CLOUD** | file store on Drive |
| 5 | Register a model version | **LOCAL only** | never in Colab - see below |
| 6 | Download `best.pt` | handoff | browser or Drive |
| 7 | Re-export ONNX / OpenVINO | **LOCAL** | RTX A1000, 4 GB |
| 8 | INT8 calibration | **LOCAL** | CPU + local footage |
| 9 | Frontier benchmark and the promotion gate | **LOCAL** | RTX A1000, 4 GB |

Step 5 is split out deliberately. The MLflow Model Registry needs a database-backed tracking server
(`sqlite:///.../mlflow.db`), which a Colab VM does not have and should not have: the thing worth
registering is the artifact that will actually serve traffic, measured on the host that will serve
it. Colab produces a `best.pt` and a run history. Promotion is a local decision.

## Running it

1. Upload `ml/scripts/train_colab.py` to Colab. It is `# %%` cell-delimited, so
   `jupytext --to notebook train_colab.py` converts it cleanly, or paste the cells in order into a
   fresh notebook. It also runs as-is with `!python train_colab.py`.
2. *Runtime > Change runtime type > T4 GPU*. The script exits immediately on a CPU runtime rather
   than starting a run that would take days.
3. Run all cells. It mounts Drive, asks for the usual authorization, and writes everything it wants
   to keep under `MyDrive/streamsight/`.

Knobs worth knowing before you start, all near the top of the script:

| Constant | Default | Why you would change it |
|---|---|---|
| `MAX_TRAIN_IMAGES` | 8000 | The main lever on wall-clock. Lower it if your measured first epoch says the budget is at risk |
| `MAX_VAL_IMAGES` | 1000 | Validation runs after every epoch; this is not free |
| `EPOCHS` / `BATCH` | 10 / 4 | Batch 4 is the PRD figure; a T4 tolerates more, but then the run is no longer the one that was scoped |
| `SAVE_PERIOD` | 5 | The FR-13 checkpoint interval. Lowering it costs Drive writes and buys back lost epochs |
| `SEED` | 0 | **Do not change mid-project.** It selects which images are in the subset, so changing it silently changes the dataset a resumed run continues against |

## Expected wall clock

These are budgets from the PRD, not observations:

| | Budget | Source |
|---|---|---|
| Whole fine-tune, 10 epochs | ~3-5 GPU-h, one session | PRD success-metrics table |
| Free Colab session limit | commonly ~12 h, no guarantee | Colab's own terms; sessions are reclaimed without warning |

The PRD's own instruction is to **measure one epoch first** and extrapolate before committing to the
rest. The script prints `session wall clock` at the end and logs it to MLflow, so after one session
you have a real number - use it, and update this table with it.

### The Colab subset is not the local subset

`ml/data/scripts/` builds a **val2017** subset locally, for INT8 calibration, parity and evaluation.
The Colab script builds its own **train2017 + val2017** subset on the VM, because training data has
no reason to travel through this laptop. Both use the same six classes in the same order - `person,
bicycle, car, motorcycle, bus, truck` - so the contiguous class ids 0-5 mean the same thing on both
sides. That ordering is a contract; changing it in one place and not the other silently mislabels
everything downstream.

Dataset build is separate from training time: ~241 MB of annotations plus one HTTP request per
selected image, 32 at a time. It is rebuilt from scratch in every new session, because `/content` is
wiped and keeping ~1 GB of images synced to Drive costs more than re-fetching them.

## When the session dies

This is the expected case, not the exception (PRD R3), so resume is the default path rather than a
flag you have to remember to pass while the session that needed it is already gone.

**What to do: re-run the notebook from the top. That is the whole procedure.**

What makes that work:

- `project=` points at `MyDrive/streamsight/runs/`, so Ultralytics writes `last.pt` there after
  every epoch and `epoch5.pt`, `epoch10.pt` at the `SAVE_PERIOD` interval. None of it lives on the
  VM.
- On start, the script checks for `runs/yolo11n-person-vehicle/weights/last.pt`. If it exists it
  loads that checkpoint and calls `model.train(resume=True)`, which restores the optimizer state,
  the LR schedule position and the epoch counter - not just the weights.
- The MLflow run id is stored in `MyDrive/streamsight/yolo11n-person-vehicle.mlflow_run_id`, so the
  resumed session appends to the same run instead of leaving a trail of 5-epoch fragments.
- The dataset is re-selected with the same seed and sorted ids, so the rebuilt subset has the same
  membership as the one the checkpoint was trained on.

Worst case you lose the epochs since the last `last.pt` write, which is at most one epoch of
compute, and at most 5 epochs if you fall back to a numbered checkpoint.

Two failure modes that are **not** covered by resume:

- **Changing `SEED`, `MAX_TRAIN_IMAGES` or the class list between sessions.** The checkpoint then
  continues against different data and different label indices. Delete the run directory and start
  over instead.
- **Renaming or deleting `runs/yolo11n-person-vehicle/`.** That directory *is* the resume state.
  Ultralytics needs its `args.yaml` alongside `last.pt`.

To start genuinely fresh, delete `MyDrive/streamsight/runs/yolo11n-person-vehicle/` and the
`.mlflow_run_id` file.

## Where `best.pt` goes

The last cell downloads `best.pt`; Drive keeps the authoritative copy at
`MyDrive/streamsight/runs/yolo11n-person-vehicle/weights/best.pt`.

Locally:

```powershell
# 1. Keep it under its own name, so provenance is not lost the moment it is activated
Copy-Item <downloaded>\best.pt ml\models\weights\yolo11n_person_vehicle.pt

# 2. The runtime loads weights/yolo11n.pt by name (apps/api/app/inference/backends.py), so activating
#    the fine-tune means shadowing the pretrained file - back it up first
Copy-Item ml\models\weights\yolo11n.pt ml\models\weights\yolo11n_pretrained.pt
Copy-Item ml\models\weights\yolo11n_person_vehicle.pt ml\models\weights\yolo11n.pt
```

Reverting is `Copy-Item ml\models\weights\yolo11n_pretrained.pt ml\models\weights\yolo11n.pt` plus a
re-export.

## How it flows into quantization and the gate

The fine-tuned model has **6 classes, not 80**. Every artifact derived from the old weights is stale
the moment you swap the file - the exported ONNX and OpenVINO graphs still carry an 80-class head,
and detections already written to `apps/api/data/stream.db` carry the old class ids. Re-export
before drawing any conclusion from a benchmark.

```powershell
# 1. Re-export against the new weights
python ml/quantization/export_engines.py --formats fp16_onnx openvino_cpu

# 2. Re-calibrate INT8 - activation ranges belong to the model that produced them
python ml/quantization/calibrate.py --frames 128

# 3. Re-measure the frontier
python ml/eval/benchmark_frontier.py --frames 200 --imgsz 640 480 320

# 4. Re-score mAP on the 6-class subset, then run the promotion gate
python ml/quantization/benchmark_precision.py --dry-run
```

Then the promotion condition, unchanged by where training happened: **INT8 mAP50-95 must be within
3% absolute of the locally-measured FP32 baseline on the same 6-class set** (PRD FR-16). Both halves
of that comparison are measured locally, on the 6-class subset. The 80-class published YOLO11n
figure (39.5% mAP50-95) is a sanity reference only and is not a valid baseline for this model - the
class sets are different, so the numbers are not comparable. The gate refuses a comparison across
mismatched class sets outright, which is the check that catches a half-finished swap: fine-tuned
weights in place, stale 80-class exports beside them.

The Colab validation metrics logged to MLflow are useful for spotting a training run that went
wrong. They are not the gate. A number produced on a T4 says nothing about the engine that will
serve on a 4 GB A1000.

See `docs/QUANTIZATION.md` for what the export step actually does and which target to pick, and
`docs/BENCHMARKS.md` for the measured frontier.

Phu Nguyen - HCMC, Vietnam
