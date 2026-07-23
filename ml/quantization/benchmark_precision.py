"""Promotion gate: decides which exported detector is allowed to be Production.

Runs **locally**, never on Colab. Colab fine-tuning produces `best.pt` and logs
params/metrics to an ephemeral tracking store; nothing there can be registered,
because the MLflow **Model Registry is not implemented on the bare `mlruns/`
file store** -- `register_model` and `transition_model_version_stage` require a
database-backed tracking server. That single constraint is the reason this repo
pins a SQLite-backed server rather than the default local directory:

    mlflow server --backend-store-uri sqlite:///D:/FPT/Demo/streamsight/mlflow.db \\
        --artifacts-destination file:///D:/FPT/Demo/streamsight/mlartifacts \\
        --serve-artifacts --host 127.0.0.1 --port 5000

`--start-server` runs exactly that command for you; `--print-server-command`
prints it. See :func:`server_command` for why the artifact root is proxied
rather than passed as ``--default-artifact-root``.

The gate reads accuracy from the JSON that ``ml/eval/eval_coco.py`` writes rather
than measuring anything itself, so promotion is decided on the same numbers a
human reads in the report -- there is no second, privately-computed mAP that
could disagree. The reader is deliberately strict: a silently-missing key would
promote an engine on a default value, so an unexpected report shape raises
:class:`GateContractError` naming the file, the keys it looked for, and the keys
it found.

Promotion rules (PRD FR-16):

* **Engine** -- the INT8 candidate goes to Production iff its mAP50-95 is within
  ``gate.map_drop_max`` *absolute* of the locally-measured FP32 baseline **on the
  same class set**. A mismatched class set is refused outright, because a 6-class
  fine-tune scored against an 80-class baseline produces a meaningless delta.
  On failure the FP16 sibling is promoted instead, so the API always has a
  Production version to load.
* **Tracking quality** -- MOT17 IDF1 is registered and gated under a *separate*
  model name. Tracking is a runtime association concern that quantization does
  not control, so a weak IDF1 must not veto a perfectly good engine.

Usage:
    python ml/quantization/benchmark_precision.py --dry-run
    python ml/quantization/benchmark_precision.py --candidate int8_onnx_cpu
    python ml/quantization/benchmark_precision.py --print-server-command
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.core.config import get_settings  # noqa: E402
from app.inference.backends import BACKENDS  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "ml" / "train" / "config.yaml"
EVAL_REPORTS_DIR = REPO_ROOT / "ml" / "eval" / "reports"
#: ``eval_coco.py`` / ``eval_mot.py`` write one file per backend+resolution, so
#: the gate collects a directory rather than opening a single fixed path.
COCO_REPORT_GLOB = "coco_*.json"
MOT_REPORT_GLOB = "mot_*.json"
DEFAULT_OUT = REPO_ROOT / "ml" / "quantization" / "reports" / "promotion.json"

#: Backing store for the tracking + registry server. A file store cannot hold a
#: registry, so this path is part of the contract rather than a preference.
MLFLOW_DB_PATH = REPO_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = REPO_ROOT / "mlartifacts"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

#: Accepted spellings of COCO mAP50-95 in the eval report. Ordered most explicit
#: first; bare ``map`` is last because it is the most ambiguous.
COCO_PRIMARY_KEYS: tuple[str, ...] = ("map50_95", "map50-95", "mAP50-95", "map_50_95", "map")
COCO_SECONDARY_KEYS: tuple[str, ...] = ("map50", "map_50", "mAP50")
#: ``eval_coco.py`` names the backend ``engine``; the other two spellings are
#: accepted so a report reshaped later still parses instead of silently missing.
BACKEND_ID_KEYS: tuple[str, ...] = ("engine", "backend", "backend_key", "name")
MOT_IDF1_KEYS: tuple[str, ...] = ("idf1", "IDF1", "idf1_overall")


class GateContractError(RuntimeError):
    """An eval report did not carry the fields the gate needs to decide."""


@dataclass
class EvalRecord:
    """One backend's accuracy as reported by ``eval_coco.py``."""

    backend: str
    map50_95: float
    map50: float | None = None
    imgsz: int | None = None
    classes: tuple[str, ...] | None = None
    #: ``eval_coco.py``'s label for the evaluated subset, e.g. ``prd6``.
    class_set: str | None = None
    source: str = ""


@dataclass
class GateDecision:
    """The engine-promotion verdict and every number that produced it."""

    baseline_backend: str
    baseline_map50_95: float
    candidate_backend: str
    candidate_map50_95: float
    absolute_drop: float
    max_absolute_drop: float
    passed: bool
    promoted_backend: str
    fallback_backend: str
    class_set: list[str] | None = None
    reason: str = ""


@dataclass
class TrackerDecision:
    """The tracking-quality verdict, gated independently of the engine."""

    idf1: float
    min_idf1: float
    passed: bool
    source: str = ""
    blocks_engine_promotion: bool = False


@dataclass
class GateReport:
    """Everything the gate did, written to disk so a run is auditable offline."""

    generated_at: str
    dry_run: bool
    tracking_uri: str
    engine: dict[str, Any]
    coco_reports: list[str] = field(default_factory=list)
    tracker: dict[str, Any] | None = None
    registry: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def load_gate_config(path: Path) -> dict[str, Any]:
    """Read thresholds from the same Hydra config the trainer consumes.

    Sharing one file is what stops the gate and the training run from drifting to
    two different definitions of "acceptable accuracy drop".
    """
    from omegaconf import OmegaConf

    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    # resolve=False: the ``hydra:`` block uses the ``${now:...}`` resolver, which
    # only exists inside a Hydra application.
    container = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(container, dict):
        raise SystemExit(f"config is not a mapping: {path}")
    return container


def _cfg(config: dict[str, Any], path: str, default: Any) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# --------------------------------------------------------------------------- #
# eval report readers
# --------------------------------------------------------------------------- #


def _records_from_payload(payload: Any, source: Path) -> list[dict[str, Any]]:
    """Normalise the plausible container shapes into a list of record dicts."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, dict):
        raise GateContractError(f"{source}: expected an object or array at the top level")

    for key in ("results", "backends", "per_backend", "evaluations"):
        node = payload.get(key)
        if isinstance(node, list):
            return [r for r in node if isinstance(r, dict)]
        if isinstance(node, dict):
            return [{"backend": k, **v} for k, v in node.items() if isinstance(v, dict)]

    # A single-record report, e.g. one backend evaluated per invocation.
    if any(k in payload for k in COCO_PRIMARY_KEYS):
        return [payload]
    raise GateContractError(
        f"{source}: no per-backend results found. Looked for a top-level array, or one of "
        f"'results'/'backends'/'per_backend'/'evaluations', or a flat single-backend record. "
        f"Top-level keys present: {sorted(payload)}"
    )


def _pick(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _classes_of(record: dict[str, Any], payload: Any) -> tuple[str, ...] | None:
    raw = record.get("classes") or record.get("class_names")
    if raw is None and isinstance(payload, dict):
        raw = payload.get("classes") or payload.get("class_names")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    return tuple(str(c) for c in raw)


def discover_reports(explicit: list[Path] | None, directory: Path, pattern: str) -> list[Path]:
    """Resolve report paths, expanding directories with ``pattern``.

    Returned oldest-first so that when two files describe the same backend the
    newer measurement is the one that survives the merge.
    """
    paths: list[Path] = []
    for candidate in explicit or [directory]:
        if candidate.is_dir():
            paths.extend(candidate.glob(pattern))
        else:
            paths.append(candidate)
    return sorted(
        {p.resolve() for p in paths}, key=lambda p: p.stat().st_mtime if p.exists() else 0
    )


def load_coco_records(paths: list[Path]) -> dict[str, EvalRecord]:
    """Parse ``eval_coco.py`` output into ``{backend: EvalRecord}``.

    Raises:
        GateContractError: if a file exists but does not carry a backend
            identifier and an mAP50-95 value per record. Defaulting here would
            silently promote an engine on a number nobody measured.
    """
    if not paths:
        raise GateContractError(
            f"no COCO eval reports found under {EVAL_REPORTS_DIR} matching '{COCO_REPORT_GLOB}'. "
            f"Run ml/eval/eval_coco.py for the baseline and each candidate, or pass "
            f"--coco-report explicitly."
        )
    records: dict[str, EvalRecord] = {}
    for path in paths:
        if not path.exists():
            raise GateContractError(f"COCO eval report not found: {path}")
        payload = _read_json(path)
        for raw in _records_from_payload(payload, path):
            backend = _pick(raw, BACKEND_ID_KEYS)
            primary = _pick(raw, COCO_PRIMARY_KEYS)
            if backend is None or primary is None:
                raise GateContractError(
                    f"{path}: a record is missing required fields. Need a backend identifier "
                    f"(one of {list(BACKEND_ID_KEYS)}) and mAP50-95 (one of "
                    f"{list(COCO_PRIMARY_KEYS)}). Record keys: {sorted(raw)}"
                )
            secondary = _pick(raw, COCO_SECONDARY_KEYS)
            imgsz = raw.get("imgsz") or raw.get("resolution")
            class_set = raw.get("class_set") or (
                payload.get("class_set") if isinstance(payload, dict) else None
            )
            records[str(backend)] = EvalRecord(
                backend=str(backend),
                map50_95=_as_fraction(float(primary)),
                map50=None if secondary is None else _as_fraction(float(secondary)),
                imgsz=None if imgsz is None else int(imgsz),
                classes=_classes_of(raw, payload),
                class_set=None if class_set is None else str(class_set),
                source=path.name,
            )
    if not records:
        raise GateContractError(f"{paths}: parsed successfully but contained zero backend records")
    return records


def load_mot_idf1(paths: list[Path]) -> tuple[float, str]:
    """Read the overall MOT17 IDF1 from ``eval_mot.py`` output.

    Returns the newest report's value with the file it came from, so the logged
    number can be traced back to a specific evaluation run.
    """
    if not paths:
        raise GateContractError(
            f"no MOT eval reports found under {EVAL_REPORTS_DIR} matching '{MOT_REPORT_GLOB}'. "
            f"Run ml/eval/eval_mot.py, or pass --skip-tracker to gate the engine only."
        )
    path = paths[-1]
    if not path.exists():
        raise GateContractError(f"MOT eval report not found: {path}")
    return _idf1_from(_read_json(path), path), path.name


def _read_json(path: Path) -> Any:
    """Load a report, tolerating a BOM and turning parse errors into gate errors.

    ``utf-8-sig`` rather than ``utf-8`` because PowerShell's default redirection
    writes a BOM on Windows, and an operator regenerating a report by hand should
    not be met with a raw ``JSONDecodeError``.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise GateContractError(f"{path}: not valid JSON ({exc})") from exc


def _idf1_from(payload: Any, path: Path) -> float:
    if isinstance(payload, dict):
        direct = _pick(payload, MOT_IDF1_KEYS)
        if direct is not None:
            return _as_fraction(float(direct))
        overall = payload.get("overall") or payload.get("summary") or payload.get("OVERALL")
        if isinstance(overall, dict):
            value = _pick(overall, MOT_IDF1_KEYS)
            if value is not None:
                return _as_fraction(float(value))
        for key in ("results", "sequences"):
            node = payload.get(key)
            rows = node if isinstance(node, list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("sequence") or row.get("name") or "").upper()
                value = _pick(row, MOT_IDF1_KEYS)
                if name in {"OVERALL", "COMBINED", "ALL"} and value is not None:
                    return _as_fraction(float(value))
    raise GateContractError(
        f"{path}: no overall IDF1 found. Looked for a top-level {list(MOT_IDF1_KEYS)}, an "
        f"'overall'/'summary' object carrying one, or an OVERALL row under 'results'/'sequences'."
    )


def _as_fraction(value: float) -> float:
    """Normalise a metric to 0-1.

    Both conventions are in circulation (pycocotools returns 0.31, report tables
    often print 31.2). Values above 1 can only be percentages, so treating them
    as such is unambiguous -- and getting this wrong would compare 0.31 against
    a 3-percentage-point threshold and pass everything.
    """
    return value / 100.0 if value > 1.0 else value


# --------------------------------------------------------------------------- #
# gate logic
# --------------------------------------------------------------------------- #


def _require_same_class_set(baseline: EvalRecord, candidate: EvalRecord) -> None:
    """Refuse to compare mAP measured on different label sets.

    The PRD is emphatic about this: the deployed detector is a 6-class fine-tune,
    and an absolute delta against an 80-class baseline is not a quantization
    signal at all -- it is mostly the class-set change. Silently allowing it
    would let a broken INT8 engine pass, or a good one fail.
    """
    for attribute, describe in (("classes", list), ("class_set", str)):
        left = getattr(baseline, attribute)
        right = getattr(candidate, attribute)
        if left is not None and right is not None and left != right:
            raise GateContractError(
                f"class-set mismatch: baseline '{baseline.backend}' ({baseline.source}) was scored "
                f"on {describe(left)} but candidate '{candidate.backend}' ({candidate.source}) on "
                f"{describe(right)}. Re-run ml/eval/eval_coco.py with the same --classes for both."
            )


def decide(
    records: dict[str, EvalRecord],
    baseline_key: str,
    candidate_key: str,
    fallback_key: str,
    max_drop: float,
) -> GateDecision:
    """Apply the quantization condition and pick the backend to promote."""
    missing = [k for k in (baseline_key, candidate_key) if k not in records]
    if missing:
        raise GateContractError(
            f"eval report has no entry for {missing}. Backends present: {sorted(records)}"
        )
    baseline = records[baseline_key]
    candidate = records[candidate_key]

    _require_same_class_set(baseline, candidate)

    drop = baseline.map50_95 - candidate.map50_95
    passed = drop <= max_drop
    promoted = candidate_key if passed else fallback_key
    if not passed and fallback_key not in records:
        raise GateContractError(
            f"INT8 candidate failed the gate (drop {drop:.4f} > {max_drop:.4f}) and the fallback "
            f"'{fallback_key}' has no eval record, so there is nothing to promote instead. "
            f"Backends present: {sorted(records)}"
        )
    reason = (
        f"INT8 mAP50-95 drop {drop * 100:.2f}pp <= {max_drop * 100:.2f}pp"
        if passed
        else f"INT8 mAP50-95 drop {drop * 100:.2f}pp > {max_drop * 100:.2f}pp; falling back to FP16"
    )
    class_set = list(baseline.classes) if baseline.classes else None
    return GateDecision(
        baseline_backend=baseline_key,
        baseline_map50_95=round(baseline.map50_95, 5),
        candidate_backend=candidate_key,
        candidate_map50_95=round(candidate.map50_95, 5),
        absolute_drop=round(drop, 5),
        max_absolute_drop=max_drop,
        passed=passed,
        promoted_backend=promoted,
        fallback_backend=fallback_key,
        class_set=class_set,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# mlflow
# --------------------------------------------------------------------------- #


def server_command() -> list[str]:
    """Argv of the DB-backed tracking + registry server this gate requires.

    Artifacts are *proxied* (``--artifacts-destination`` + the default
    ``mlflow-artifacts:/`` root) rather than addressed with
    ``--default-artifact-root`` as the PRD's example line does. Measured on this
    machine: an absolute Windows root is stored back as ``d:/FPT/...``, and
    MLflow then resolves artifact repositories by URI scheme, reading ``d`` as
    the scheme -- every ``log_artifact`` dies with "could not find a registered
    artifact repository" *after* the run row already exists. A relative root
    dodges that only while every client shares the server's working directory.
    Proxying makes the client-side URI ``mlflow-artifacts:/``, which carries no
    drive letter at all.
    """
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
        "--artifacts-destination",
        MLFLOW_ARTIFACT_ROOT.as_uri(),
        "--serve-artifacts",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
    ]


def start_server() -> int:
    """Run the tracking server in the foreground until interrupted."""
    command = server_command()
    print(" ".join(command))
    MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.call(command)  # noqa: S603 - fixed argv, no shell
    except KeyboardInterrupt:
        return 0


def _artifact_for(backend_key: str) -> Path:
    settings = get_settings()
    backend = BACKENDS.get(backend_key)
    if backend is None:
        raise SystemExit(f"unknown backend '{backend_key}'. Known: {sorted(BACKENDS)}")
    return backend.path(settings)


def _log_backend_run(
    mlflow: Any,
    backend_key: str,
    record: EvalRecord,
    decision: GateDecision,
) -> str:
    """Log one candidate as a run, attach its artifact, and register a version."""
    artifact = _artifact_for(backend_key)
    if not artifact.exists():
        raise SystemExit(
            f"cannot register '{backend_key}': artifact missing at {artifact}. "
            f"Export it before running the gate."
        )
    with mlflow.start_run(run_name=f"gate-{backend_key}") as run:
        mlflow.log_params(
            {
                "backend": backend_key,
                "label": BACKENDS[backend_key].label,
                "device": BACKENDS[backend_key].device,
                "imgsz": record.imgsz or BACKENDS[backend_key].export_imgsz,
                "artifact": artifact.name,
                "baseline_backend": decision.baseline_backend,
                "max_absolute_drop": decision.max_absolute_drop,
                "class_set": ",".join(decision.class_set or []),
                "eval_report": record.source,
            }
        )
        metrics = {
            "map50_95": record.map50_95,
            "absolute_drop_vs_fp32": decision.baseline_map50_95 - record.map50_95,
            "artifact_mb": round(_size_mb(artifact), 3),
        }
        if record.map50 is not None:
            metrics["map50"] = record.map50
        mlflow.log_metrics(metrics)
        if artifact.is_dir():
            mlflow.log_artifacts(str(artifact), artifact_path="model")
        else:
            mlflow.log_artifact(str(artifact), artifact_path="model")
        return run.info.run_id


def _size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2
    return path.stat().st_size / 1024**2


def promote_engine(
    tracking_uri: str,
    experiment: str,
    registry_model: str,
    records: dict[str, EvalRecord],
    decision: GateDecision,
) -> dict[str, Any]:
    """Register both candidates and transition the winner to Production."""
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    client = MlflowClient(tracking_uri=tracking_uri)

    versions: dict[str, str] = {}
    candidates = [decision.candidate_backend]
    if decision.fallback_backend in records:
        candidates.append(decision.fallback_backend)

    for backend_key in candidates:
        run_id = _log_backend_run(mlflow, backend_key, records[backend_key], decision)
        try:
            version = mlflow.register_model(f"runs:/{run_id}/model", registry_model)
        except MlflowException as exc:
            raise SystemExit(_registry_hint(exc, tracking_uri)) from exc
        versions[backend_key] = version.version

    promoted_version = versions[decision.promoted_backend]
    try:
        client.transition_model_version_stage(
            name=registry_model,
            version=promoted_version,
            stage="Production",
            archive_existing_versions=True,
        )
    except MlflowException as exc:
        raise SystemExit(_registry_hint(exc, tracking_uri)) from exc

    return {
        "model": registry_model,
        "versions": versions,
        "production_version": promoted_version,
        "production_backend": decision.promoted_backend,
    }


def promote_tracker(
    tracking_uri: str,
    experiment: str,
    registry_model: str,
    tracker: TrackerDecision,
    detector_backend: str,
) -> dict[str, Any]:
    """Log and gate tracking quality under its own registry name.

    Kept in a separate model because IDF1 measures association over time, which
    quantization does not determine; letting it archive the detector would take
    the API offline for a defect it cannot fix by swapping engines.
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    client = MlflowClient(tracking_uri=tracking_uri)

    tracker_config = get_settings().tracker_config_path
    with mlflow.start_run(run_name="gate-tracker-quality") as run:
        mlflow.log_params({"detector_backend": detector_backend, "min_idf1": tracker.min_idf1})
        mlflow.log_metrics({"mot17_idf1": tracker.idf1})
        if tracker_config.exists():
            mlflow.log_artifact(str(tracker_config), artifact_path="model")
        run_id = run.info.run_id

    try:
        version = mlflow.register_model(f"runs:/{run_id}/model", registry_model)
        if tracker.passed:
            client.transition_model_version_stage(
                name=registry_model,
                version=version.version,
                stage="Production",
                archive_existing_versions=True,
            )
    except MlflowException as exc:
        raise SystemExit(_registry_hint(exc, tracking_uri)) from exc

    return {
        "model": registry_model,
        "version": version.version,
        "stage": "Production" if tracker.passed else "None",
    }


def _registry_hint(exc: Exception, tracking_uri: str) -> str:
    return (
        f"MLflow registry call failed against {tracking_uri}: {exc}\n"
        f"The Model Registry needs a database-backed server. Start one with:\n"
        f"  {' '.join(server_command())}\n"
        f"then set MLFLOW_TRACKING_URI={DEFAULT_TRACKING_URI}"
    )


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidate", default="int8_onnx_cpu", help="INT8 backend under test")
    parser.add_argument("--baseline", default="fp32_gpu", help="locally-measured FP32 reference")
    parser.add_argument("--fallback", default="fp16_onnx", help="promoted when the candidate fails")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--coco-report",
        type=Path,
        action="append",
        help=(
            f"eval_coco.py report file or directory; repeatable. "
            f"Default: {EVAL_REPORTS_DIR}/{COCO_REPORT_GLOB}"
        ),
    )
    parser.add_argument(
        "--mot-report",
        type=Path,
        action="append",
        help=f"eval_mot.py report file or directory. Default: {EVAL_REPORTS_DIR}/{MOT_REPORT_GLOB}",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-tracker", action="store_true", help="gate the engine only")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate the gate and write the report without contacting MLflow",
    )
    parser.add_argument("--print-server-command", action="store_true")
    parser.add_argument("--start-server", action="store_true")
    args = parser.parse_args(argv)

    if args.print_server_command:
        print(" ".join(server_command()))
        return 0
    if args.start_server:
        return start_server()

    config = load_gate_config(args.config)
    max_drop = float(_cfg(config, "gate.map_drop_max", 0.03))
    min_idf1 = float(_cfg(config, "gate.mot_idf1_min", 0.60))
    experiment = str(_cfg(config, "mlflow.experiment", "streamsight"))
    detector_model = str(_cfg(config, "mlflow.registry_model", "streamsight-detector"))
    tracker_model = str(_cfg(config, "mlflow.tracker_quality_model", "streamsight-tracker-quality"))
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", str(_cfg(config, "mlflow.tracking_uri", DEFAULT_TRACKING_URI))
    )

    coco_paths = discover_reports(args.coco_report, EVAL_REPORTS_DIR, COCO_REPORT_GLOB)
    try:
        records = load_coco_records(coco_paths)
        decision = decide(records, args.baseline, args.candidate, args.fallback, max_drop)
    except GateContractError as exc:
        print(f"gate contract violated: {exc}", file=sys.stderr)
        return 2

    tracker: TrackerDecision | None = None
    if not args.skip_tracker:
        try:
            idf1, mot_source = load_mot_idf1(
                discover_reports(args.mot_report, EVAL_REPORTS_DIR, MOT_REPORT_GLOB)
            )
        except GateContractError as exc:
            print(f"tracker gate skipped: {exc}", file=sys.stderr)
        else:
            tracker = TrackerDecision(
                idf1=idf1, min_idf1=min_idf1, passed=idf1 >= min_idf1, source=mot_source
            )

    print(f"reports   {', '.join(p.name for p in coco_paths)}")
    print(f"baseline  {decision.baseline_backend:<16} mAP50-95 {decision.baseline_map50_95:.4f}")
    print(f"candidate {decision.candidate_backend:<16} mAP50-95 {decision.candidate_map50_95:.4f}")
    print(f"drop      {decision.absolute_drop * 100:.2f}pp (limit {max_drop * 100:.2f}pp)")
    print(f"verdict   {'PASS' if decision.passed else 'FAIL'} - {decision.reason}")
    print(f"promote   {decision.promoted_backend}")
    if tracker is not None:
        state = "PASS" if tracker.passed else "FAIL"
        print(
            f"tracker   MOT17 IDF1 {tracker.idf1:.4f} (min {min_idf1:.2f}) {state} - advisory only"
        )

    report = GateReport(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        dry_run=args.dry_run,
        tracking_uri=tracking_uri,
        engine=asdict(decision),
        coco_reports=[str(p) for p in coco_paths],
        tracker=None if tracker is None else asdict(tracker),
    )

    if not args.dry_run:
        report.registry["detector"] = promote_engine(
            tracking_uri, experiment, detector_model, records, decision
        )
        if tracker is not None:
            report.registry["tracker_quality"] = promote_tracker(
                tracking_uri, experiment, tracker_model, tracker, decision.promoted_backend
            )
        print(
            f"registered {detector_model} -> Production version "
            f"{report.registry['detector']['production_version']}"
        )
    else:
        print("dry run: nothing registered, no MLflow server contacted")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
