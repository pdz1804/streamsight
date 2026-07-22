# %% [markdown]
# # StreamSight - Colab fine-tuning (FR-13)
#
# **This script has never been executed by its author.** It is written to be run **by you, in
# Google Colab, on your own account**, because training is explicitly cloud-only (PRD NG1/C1: the
# 4 GB laptop GPU that serves inference cannot train). Nothing below has been validated on a T4,
# and no number in this file is a measurement - every duration is a budget from the PRD, not an
# observation. Record the real ones in `docs/TRAINING_GUIDE.md` after your first run.
#
# Open it in Colab via *File > Upload notebook* after converting (`jupytext --to notebook
# train_colab.py`), or paste the `# %%` cells in order into a fresh notebook. It also runs
# unmodified as a plain script inside a Colab cell (`!python train_colab.py`).
#
# What it does: fine-tunes pretrained `yolo11n.pt` on the COCO person+vehicle 6-class subset for
# ~10 epochs at batch 4, checkpointing to Drive so a killed session loses at most 5 epochs, and
# logging params/metrics to an MLflow file store on Drive. It deliberately does **not** register
# models - registration and the promotion gate are local, against the DB-backed registry (FR-16).

# %%
"""Colab fine-tuning entry point for the StreamSight detector.

Resume is the default path rather than a flag because the failure this guards against - a free
Colab VM being reclaimed mid-run - is silent and common, and a flag that has to be remembered
after the session is already gone is not a mitigation (PRD R3). Every run therefore looks for the
previous run directory on Drive first and only starts fresh when there is nothing to continue.

Checkpoints and the MLflow store live on Drive, not on the VM, for the same reason: `/content` is
wiped when the session ends. The dataset does not, because rebuilding it is deterministic and
costs bandwidth rather than correctness, while keeping ~1 GB of images in sync with Drive costs
more time than re-downloading them.
"""

# Cells install their dependencies before importing them, so imports cannot all sit at the top.
# ruff: noqa: E402

import subprocess
import sys

#: Pinned to the versions the rest of the repo is validated against (`requirements.txt`), so a
#: checkpoint trained here loads in the local runtime without a version negotiation.
PINNED_DEPS = [
    "ultralytics==8.4.104",
    "mlflow==2.17.0",
    "pycocotools==2.0.8",
]


def pip_install(packages: list[str]) -> None:
    """Install into the Colab runtime.

    torch is deliberately not reinstalled: Colab's build is matched to its own driver, and
    replacing it is the most common way to lose the GPU mid-session.
    """
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True,
    )


pip_install(PINNED_DEPS)

# %%
import json
import random
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# %% [markdown]
# ## Configuration
#
# `DRIVE_PROJECT` is the only thing that survives a session. Keep the name stable: it is what makes
# the resume path find the previous attempt.

# %%
DRIVE_MOUNT = Path("/content/drive")
DRIVE_PROJECT = DRIVE_MOUNT / "MyDrive" / "streamsight"
RUNS_DIR = DRIVE_PROJECT / "runs"
RUN_NAME = "yolo11n-person-vehicle"
MLFLOW_DIR = DRIVE_PROJECT / "mlruns"
RUN_ID_FILE = DRIVE_PROJECT / f"{RUN_NAME}.mlflow_run_id"

WORK_DIR = Path("/content/streamsight")
DATASET_DIR = WORK_DIR / "coco_person_vehicle"
ANNOTATION_DIR = WORK_DIR / "coco_annotations"

EPOCHS = 10
BATCH = 4
IMGSZ = 640
#: PRD FR-13: a checkpoint every 5 epochs, so a lost session costs at most 5 epochs of compute.
SAVE_PERIOD = 5
BASE_WEIGHTS = "yolo11n.pt"
SEED = 0

#: COCO category ids (1-indexed, with gaps) mapped to the contiguous class ids the deployed model
#: uses. Order fixes the label indices, so it must not be reshuffled between runs or a resumed run
#: would train against relabelled data.
COCO_CATEGORY_IDS = {
    1: 0,  # person
    2: 1,  # bicycle
    3: 2,  # car
    4: 3,  # motorcycle
    6: 4,  # bus
    8: 5,  # truck
}
CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

#: Caps on the subset. The PRD budgets ~3-5 GPU-h for the whole fine-tune on a free T4; these are
#: the knob that buys that budget back if your measured first epoch says otherwise. Selection is
#: seeded, so lowering them mid-project changes the data and invalidates a resume.
MAX_TRAIN_IMAGES = 8000
MAX_VAL_IMAGES = 1000
DOWNLOAD_THREADS = 32

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# %% [markdown]
# ## Mount Drive and check the GPU
#
# Stop here if Colab handed you a CPU runtime: *Runtime > Change runtime type > T4 GPU*. Training
# on the CPU runtime would run for days rather than hours.

# %%
from google.colab import drive  # type: ignore[import-not-found]

drive.mount(str(DRIVE_MOUNT))
DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

import torch

if not torch.cuda.is_available():
    raise SystemExit("No GPU attached. Runtime > Change runtime type > T4 GPU, then re-run.")
print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

# %% [markdown]
# ## Build the 6-class subset
#
# Images are fetched individually from their COCO URLs instead of pulling `train2017.zip`, because
# the zip is ~19 GB of which this subset needs a small fraction, and a Colab session that spends an
# hour downloading is an hour of the training budget. Selection is seeded and sorted, so a rebuilt
# dataset in a later session is byte-identical in membership - which is what makes resume safe.

# %%
from pycocotools.coco import COCO


def download_annotations() -> Path:
    """Fetch and extract the COCO annotation archive once per session."""
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    train_json = ANNOTATION_DIR / "annotations" / "instances_train2017.json"
    if train_json.exists():
        return train_json.parent

    archive = ANNOTATION_DIR / "annotations_trainval2017.zip"
    if not archive.exists():
        print("downloading COCO annotations (~241 MB)")
        urllib.request.urlretrieve(ANNOTATIONS_URL, archive)  # noqa: S310
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(ANNOTATION_DIR)
    archive.unlink()
    return train_json.parent


def select_image_ids(coco: COCO, limit: int) -> list[int]:
    """Pick images containing at least one target class, deterministically."""
    image_ids: set[int] = set()
    for category_id in COCO_CATEGORY_IDS:
        image_ids.update(coco.getImgIds(catIds=[category_id]))
    ordered = sorted(image_ids)
    random.Random(SEED).shuffle(ordered)  # noqa: S311 - dataset sampling, not a security context
    return sorted(ordered[:limit])


def write_label(coco: COCO, image_id: int, meta: dict, label_path: Path) -> None:
    """Write one YOLO label file: class cx cy w h, normalized, remapped to contiguous ids."""
    width, height = meta["width"], meta["height"]
    lines: list[str] = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)):
        class_id = COCO_CATEGORY_IDS.get(ann["category_id"])
        if class_id is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        lines.append(
            f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )
    label_path.write_text("\n".join(lines), encoding="utf-8")


def fetch_image(url: str, destination: Path) -> None:
    """Download one image, skipping work already done by an earlier session."""
    if destination.exists() and destination.stat().st_size > 0:
        return
    try:
        urllib.request.urlretrieve(url, destination)  # noqa: S310
    except Exception as exc:  # noqa: BLE001 - one bad image must not abort a 8000-image build
        print(f"  skipped {destination.name}: {exc}")


def build_split(annotation_dir: Path, split: str, limit: int) -> int:
    """Materialize one YOLO-format split and return how many images it holds."""
    coco = COCO(str(annotation_dir / f"instances_{split}2017.json"))
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_ids = select_image_ids(coco, limit)
    metas = coco.loadImgs(image_ids)
    for image_id, meta in zip(image_ids, metas, strict=True):
        write_label(coco, image_id, meta, label_dir / f"{Path(meta['file_name']).stem}.txt")

    started = time.time()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        for meta in metas:
            pool.submit(fetch_image, meta["coco_url"], image_dir / meta["file_name"])
    print(f"  {split}: {len(metas)} images in {time.time() - started:.0f}s")
    return len(metas)


def write_dataset_yaml() -> Path:
    """Emit the Ultralytics dataset descriptor the trainer reads."""
    path = DATASET_DIR / "data.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    path.write_text(
        f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return path


annotation_dir = download_annotations()
build_split(annotation_dir, "train", MAX_TRAIN_IMAGES)
build_split(annotation_dir, "val", MAX_VAL_IMAGES)
DATA_YAML = write_dataset_yaml()
print(DATA_YAML.read_text(encoding="utf-8"))

# %% [markdown]
# ## MLflow
#
# Params and metrics only. The Model Registry needs a database-backed tracking server, which a
# Colab VM does not have and which the promotion gate owns locally anyway (FR-16): the cloud half
# of this pipeline produces a `best.pt` and a run history, and nothing else.
#
# The run id is written to Drive so a resumed session appends to the same MLflow run instead of
# leaving a graveyard of 5-epoch fragments.

# %%
import mlflow

mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("streamsight-colab-finetune")

existing_run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip() if RUN_ID_FILE.exists() else None
active_run = mlflow.start_run(run_id=existing_run_id) if existing_run_id else mlflow.start_run()
RUN_ID_FILE.write_text(active_run.info.run_id, encoding="utf-8")
print(f"MLflow run: {active_run.info.run_id} (resumed: {existing_run_id is not None})")

# %% [markdown]
# ## Train, resuming by default
#
# `last.pt` under the Drive run directory is the resume signal. Ultralytics reconstructs the
# optimizer state, the epoch counter and the original arguments from that checkpoint, which is why
# the run directory is not deleted or renamed between sessions.

# %%
from ultralytics import YOLO
from ultralytics import settings as ultralytics_settings

# Ultralytics' own MLflow integration logs on its own schedule and would double-write into the
# same run; the callbacks below are explicit about what leaves this notebook.
ultralytics_settings.update({"mlflow": False})

RUN_DIR = RUNS_DIR / RUN_NAME
LAST_CHECKPOINT = RUN_DIR / "weights" / "last.pt"
resuming = LAST_CHECKPOINT.exists()


def log_epoch(trainer) -> None:
    """Mirror each epoch's validation metrics into MLflow as they are produced."""
    metrics = {k.replace("(B)", "").replace("/", "_"): v for k, v in trainer.metrics.items()}
    mlflow.log_metrics(metrics, step=int(trainer.epoch))


mlflow.log_params(
    {
        "base_weights": BASE_WEIGHTS,
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "save_period": SAVE_PERIOD,
        "classes": ",".join(CLASS_NAMES),
        "train_images": MAX_TRAIN_IMAGES,
        "val_images": MAX_VAL_IMAGES,
        "seed": SEED,
        "resumed": resuming,
    }
)

model = YOLO(str(LAST_CHECKPOINT)) if resuming else YOLO(BASE_WEIGHTS)
model.add_callback("on_fit_epoch_end", log_epoch)

started = time.time()
if resuming:
    print(f"resuming from {LAST_CHECKPOINT}")
    results = model.train(resume=True)
else:
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=0,
        seed=SEED,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        save_period=SAVE_PERIOD,
        pretrained=True,
        val=True,
    )
wall_clock_s = time.time() - started
mlflow.log_metric("session_wall_clock_s", wall_clock_s)
print(f"session wall clock: {wall_clock_s / 3600:.2f} h")

# %% [markdown]
# ## Final validation and provenance
#
# The metrics logged here are the ones to compare against the local FP32 baseline. They are *not*
# the promotion gate: the gate re-measures on the local host with `eval_coco.py`, because a number
# produced on a T4 says nothing about the engine that will actually serve.

# %%
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"

validation = YOLO(str(BEST_WEIGHTS)).val(data=str(DATA_YAML), imgsz=IMGSZ, device=0)
final_metrics = {
    "final_map50_95": float(validation.box.map),
    "final_map50": float(validation.box.map50),
    "best_pt_mb": BEST_WEIGHTS.stat().st_size / 1e6,
}
mlflow.log_metrics(final_metrics)

manifest = {
    "run_id": active_run.info.run_id,
    "run_dir": str(RUN_DIR),
    "classes": CLASS_NAMES,
    "epochs": EPOCHS,
    "batch": BATCH,
    "imgsz": IMGSZ,
    "train_images": MAX_TRAIN_IMAGES,
    "val_images": MAX_VAL_IMAGES,
    "seed": SEED,
    "metrics": final_metrics,
}
(DRIVE_PROJECT / f"{RUN_NAME}.manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
mlflow.end_run()
print(json.dumps(manifest, indent=2))

# %% [markdown]
# ## Take `best.pt` home
#
# The download below is the handoff. Locally:
#
# 1. Keep the fine-tuned file under its own name for provenance:
#    `ml/models/weights/yolo11n_person_vehicle.pt`
# 2. The runtime loads `ml/models/weights/yolo11n.pt` by name (`apps/api/app/backends.py`), so
#    activating the fine-tune means backing up the pretrained file and copying over it:
#
#    ```powershell
#    Copy-Item ml/models/weights/yolo11n.pt ml/models/weights/yolo11n_pretrained.pt
#    Copy-Item ml/models/weights/yolo11n_person_vehicle.pt ml/models/weights/yolo11n.pt
#    ```
#
# 3. Re-export and re-quantize against the new weights, then run the gate. The exports on disk
#    still belong to the 80-class model until you do:
#
#    ```powershell
#    python ml/quantization/export_engines.py --formats fp16_onnx openvino_cpu
#    python ml/quantization/calibrate.py --frames 128
#    ```
#
# Class ids change from 80 to 6, so anything cached against the old ids (exported engines, stored
# detections in `apps/api/data/stream.db`) is stale. `docs/TRAINING_GUIDE.md` has the full sequence.

# %%
shutil.copy(BEST_WEIGHTS, WORK_DIR / "best.pt")
try:
    # The browser download is a convenience; Drive is the authoritative copy.
    from google.colab import files  # type: ignore[import-not-found]

    files.download(str(WORK_DIR / "best.pt"))
except Exception as exc:  # noqa: BLE001
    print(f"browser download unavailable ({exc}); take it from {BEST_WEIGHTS}")

print(f"best.pt also on Drive at: {BEST_WEIGHTS}")
print(f"MLflow store on Drive at: {MLFLOW_DIR}")

# Phu Nguyen - HCMC, Vietnam
