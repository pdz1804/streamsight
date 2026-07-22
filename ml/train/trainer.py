"""Local-side training entry point (Hydra-configured), called by the Colab notebook.

Training is cloud-only by design -- a 4 GB laptop cannot hold a YOLO11n
fine-tune's activations at 640 px alongside the optimizer state (PRD NG1). This
module is nevertheless kept in the repo rather than inline in the notebook so
that the hyperparameters live under version control and are the *same* values
the promotion gate reads: both sides load ``ml/train/config.yaml``.

What runs where (PRD FR-16):

* **Colab** executes this file, logs params + per-epoch metrics to its own
  ephemeral MLflow store, checkpoints to Drive every ``train.checkpoint_every``
  epochs, and produces ``best.pt``. It registers nothing -- the Model Registry
  needs the local database-backed server, which Colab cannot reach.
* **Locally** you download ``best.pt``, quantize, evaluate, and then
  ``ml/quantization/benchmark_precision.py`` does the registration and the stage
  transition.

Resume exists because Colab reclaims sessions without warning. Ultralytics can
restore optimizer state and the epoch counter from ``last.pt``, so a reclaimed
session costs at most one checkpoint interval instead of the whole run.

Heavy imports are deliberately function-local: this module must import (and
``--dry-run``) on a machine with no GPU, no dataset, and no ultralytics.

Usage:
    python ml/train/trainer.py --dry-run
    python ml/train/trainer.py train.epochs=20 train.batch=8
    python ml/train/trainer.py train.resume=true
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger("streamsight.train")


def resolve(path_like: str) -> Path:
    """Interpret a config path against the repo root, not the process cwd.

    Hydra is configured with ``job.chdir: false``, but Colab still invokes this
    from arbitrary directories; anchoring here keeps one config valid everywhere.
    """
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def checkpoint_path(cfg: DictConfig) -> Path:
    """Where Ultralytics writes ``last.pt`` for this run name."""
    return resolve(cfg.train.project) / str(cfg.train.name) / "weights" / "last.pt"


def resolve_resume(cfg: DictConfig) -> Path | None:
    """Return the checkpoint to resume from, or ``None`` for a fresh run.

    Asking to resume with no checkpoint present is treated as a fresh start
    rather than an error: that is exactly the state of the first Colab session,
    and failing there would make the notebook's single command non-idempotent.
    """
    if not bool(cfg.train.resume):
        return None
    candidate = checkpoint_path(cfg)
    if candidate.exists():
        return candidate
    logger.warning("resume requested but %s does not exist - starting fresh", candidate)
    return None


def build_train_kwargs(cfg: DictConfig, resume_from: Path | None) -> dict[str, Any]:
    """Translate the Hydra config into Ultralytics ``model.train`` arguments."""
    return {
        "data": str(resolve(cfg.data.yaml)),
        "epochs": int(cfg.train.epochs),
        "batch": int(cfg.train.batch),
        "imgsz": int(cfg.model.imgsz),
        "lr0": float(cfg.train.lr0),
        "lrf": float(cfg.train.lrf),
        "optimizer": str(cfg.train.optimizer),
        "patience": int(cfg.train.patience),
        "workers": int(cfg.train.workers),
        "device": cfg.train.device,
        "seed": int(cfg.seed),
        # Ultralytics' own name for "checkpoint every N epochs".
        "save_period": int(cfg.train.checkpoint_every),
        "project": str(resolve(cfg.train.project)),
        "name": str(cfg.train.name),
        "exist_ok": True,
        "resume": resume_from is not None,
    }


def attach_mlflow(model: Any, cfg: DictConfig) -> None:
    """Log params once and metrics per epoch, without touching the registry.

    Tracking failures are downgraded to a warning: losing the metric stream is
    annoying, losing 3-5 GPU-h of Colab time because a server was unreachable is
    not acceptable.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.mlflow.experiment))
        mlflow.start_run(run_name=f"train-{cfg.train.name}")
        mlflow.log_params(
            {
                "weights": str(cfg.model.weights),
                "imgsz": int(cfg.model.imgsz),
                "epochs": int(cfg.train.epochs),
                "batch": int(cfg.train.batch),
                "lr0": float(cfg.train.lr0),
                "seed": int(cfg.seed),
                "classes": ",".join(str(c) for c in cfg.data.classes),
                "data_yaml": str(cfg.data.yaml),
            }
        )
    except Exception as exc:  # noqa: BLE001 - tracking must never abort a cloud run
        logger.warning("MLflow tracking disabled: %s", exc)
        return

    def on_epoch_end(trainer: Any) -> None:
        metrics = {k.replace("/", "_"): float(v) for k, v in (trainer.metrics or {}).items()}
        metrics["epoch_time_s"] = float(getattr(trainer, "epoch_time", 0.0) or 0.0)
        try:
            mlflow.log_metrics(metrics, step=int(trainer.epoch))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow metric log failed at epoch %s: %s", trainer.epoch, exc)

    model.add_callback("on_fit_epoch_end", on_epoch_end)


def run(cfg: DictConfig) -> dict[str, Any]:
    """Execute (or describe, under ``dry_run``) one fine-tuning run."""
    resume_from = resolve_resume(cfg)
    kwargs = build_train_kwargs(cfg, resume_from)
    weights = resume_from or resolve(f"ml/models/weights/{cfg.model.weights}")

    summary: dict[str, Any] = {
        "weights": str(weights),
        "resumed_from": str(resume_from) if resume_from else None,
        "train_kwargs": kwargs,
        "gate": OmegaConf.to_container(cfg.gate, resolve=True),
    }

    if bool(cfg.dry_run):
        logger.info("dry run - no training performed")
        print(OmegaConf.to_yaml(cfg))
        print(f"would train from: {weights}")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
        summary["status"] = "dry-run"
        return summary

    if not weights.exists():
        raise SystemExit(f"weights not found: {weights} (run ml/scripts/fetch_assets.py)")
    if not Path(kwargs["data"]).exists():
        raise SystemExit(
            f"dataset yaml not found: {kwargs['data']}\n"
            "Build the 6-class training set first:\n"
            "  python ml/data/scripts/download_coco.py            # val2017 + annotations\n"
            "  python ml/data/scripts/prepare_coco_subset.py      # filter to the 6 PRD classes\n"
            "  python ml/data/scripts/split_dataset.py            # writes the yaml + splits\n"
            "Note: split_dataset produces calibration and parity splits from val2017 (550 images), "
            "which is enough to exercise this path but NOT a real training set. A genuine "
            "fine-tune uses train2017 and runs on Colab -- see ml/scripts/train_colab.py, which "
            "builds its own dataset on the VM."
        )

    from ultralytics import YOLO

    model = YOLO(str(weights))
    if bool(cfg.mlflow.enabled):
        attach_mlflow(model, cfg)

    results = model.train(**kwargs)
    best = resolve(cfg.train.project) / str(cfg.train.name) / "weights" / "best.pt"
    summary["status"] = "trained"
    summary["best"] = str(best)
    summary["results_dir"] = str(getattr(results, "save_dir", ""))
    print(f"best weights: {best}")
    print(
        "download best.pt locally, then run ml/quantization/benchmark_precision.py to register it"
    )
    return summary


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(cfg)


def _translate_dry_run_flag(argv: list[str]) -> list[str]:
    """Accept ``--dry-run`` alongside Hydra's ``key=value`` override syntax.

    Every other script in this repo spells it ``--dry-run``; requiring
    ``dry_run=true`` here only because Hydra owns argv would be a gratuitous
    inconsistency for the operator.
    """
    return ["dry_run=true" if arg == "--dry-run" else arg for arg in argv]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_translate_dry_run_flag(sys.argv[1:])]
    main()
