"""Optional MLflow-backed artifact resolution (PRD FR-16's closing clause).

``Backend.path(settings)`` (`app.backends`) is the declarative table's only way
to turn an entry into a location, and it must keep meaning exactly that: the
promotion gate (``ml/quantization/benchmark_precision.py``) calls it to find
the *local* export it is about to upload, so redefining it to consult the
registry would make the gate upload whatever the registry last returned
instead of the file it just built.

This module adds a second, optional path used only where the runtime is about
to *load* a model for serving: :func:`resolved_backend` returns a ``Backend``
whose ``.path()`` resolves to a registry-cached artifact when one is
configured and available, and to the unmodified ``backend.path(settings)``
otherwise. The override works by pointing the copy's ``artifact`` field at an
absolute cache path -- ``Backend.path()`` joins it with ``models_dir``
unchanged, and an absolute right-hand operand wins that join on both POSIX and
Windows, so no other code needs to know resolution happened.

Strictly opt-in: nothing in this module costs an import or a network call
until ``settings.mlflow_tracking_uri`` is set. ``mlflow`` itself is imported
lazily, inside :func:`_resolve_from_registry`, so the default path -- and CI,
which never sets a tracking URI -- pays nothing for a dependency it does not
use.

Fail-soft, never silent: a dead registry must not block boot, so every
resolution failure (unreachable server, no Production version, an artifact
logged for a different backend) falls back to ``backend.path(settings)``. But
the fallback always logs a warning and records itself in
:func:`last_resolution_source`, because a *silent* fallback would let the API
claim it serves the promoted model while actually serving something else --
exactly the class of quiet-wrongness a silently tracker-routed mAP measurement
caused elsewhere in this project.

Scope limit: this makes the API registry-aware only for the artifact this
backend already understands (a file Ultralytics can load, or an OpenVINO IR
directory). It requires the ladder's own ``availability()`` check to already
consider the backend runnable -- today that means a local artifact exists --
so resolution swaps in a *different version* of that backend's artifact; it
does not resurrect a backend with no local export at all.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..core.config import Settings
from .backends import Backend

logger = logging.getLogger(__name__)

#: Registry HTTP calls must never hold up API startup or a hot-swap request.
#: MLflow's own default is 120s; ``setdefault`` overrides that only when the
#: operator has not already set the variable themselves.
_REQUEST_TIMEOUT_S = "5"

#: Where the currently loaded artifact for each backend key last came from --
#: "registry" or "path" -- kept for the startup log (and any future /metrics
#: field) to surface, so it is always observable which one is actually live.
_last_source: dict[str, str] = {}

#: How long an unreachable registry is left alone before it is tried again.
#: Long enough that one dead server cannot make the multi-rung startup ladder
#: pay a timeout per backend; short enough that a server restarted during a
#: session is picked up without restarting the API.
_UNREACHABLE_COOLDOWN_S = 300.0

#: Monotonic deadline until which the registry is presumed unreachable, or
#: ``None`` when it is not. Only *transport* failures set this. A bad model
#: name or a deleted run is a configuration error: it will fail identically on
#: every retry, costs nothing to re-ask, and must not disable a healthy
#: registry for everything else.
_unreachable_until: float | None = None


def reset_registry_state() -> None:
    """Forget cached resolution state. For tests, which must not leak into each other."""
    global _unreachable_until
    _unreachable_until = None
    _last_source.clear()


def _registry_is_sleeping() -> bool:
    return _unreachable_until is not None and time.monotonic() < _unreachable_until


def last_resolution_source(backend_key: str) -> str:
    """Return where ``backend_key`` last resolved from.

    ``"path"`` covers three cases that all mean the same thing to an
    operator -- this is the on-disk artifact, not something the registry
    vouched for: MLflow unconfigured, the registry unreachable, or a format
    mismatch. ``"registry"`` means the bytes came from the Production version
    of the configured model.
    """
    return _last_source.get(backend_key, "path")


def resolved_backend(backend: Backend, settings: Settings) -> Backend:
    """Return ``backend``, or a copy pointed at its registry-resolved artifact.

    Safe to call unconditionally regardless of configuration: with no
    tracking URI set this makes no import and no network call, and returns
    ``backend`` itself (not a copy) unchanged.
    """
    resolved = resolve_artifact(backend, settings)
    if resolved == backend.path(settings):
        return backend
    return dataclasses.replace(backend, artifact=str(resolved))


def resolve_artifact(backend: Backend, settings: Settings) -> Path:
    """Resolve the artifact location for ``backend``, preferring the registry.

    See the module docstring for the full fallback contract. Never raises:
    any registry failure is caught, logged, and answered with
    ``backend.path(settings)``.
    """
    global _unreachable_until

    default_path = backend.path(settings)
    if not settings.mlflow_tracking_uri or _registry_is_sleeping():
        _last_source[backend.key] = "path"
        return default_path

    try:
        resolved = _resolve_from_registry(backend, settings)
    except Exception as exc:  # noqa: BLE001 - any registry failure means "use the default"
        # OSError covers the transport failures -- refused connections, DNS,
        # socket timeouts -- where retrying immediately would just buy another
        # timeout. Anything else is deterministic and cheap to re-ask.
        if isinstance(exc, OSError):
            _unreachable_until = time.monotonic() + _UNREACHABLE_COOLDOWN_S
            logger.warning(
                "mlflow unreachable for backend %s (%s); serving %s instead, and "
                "leaving the registry alone for %.0fs",
                backend.key,
                _short(exc),
                default_path,
                _UNREACHABLE_COOLDOWN_S,
            )
        else:
            logger.warning(
                "mlflow resolution failed for backend %s (%s); serving %s instead",
                backend.key,
                _short(exc),
                default_path,
            )
        _last_source[backend.key] = "path"
        return default_path

    if resolved is None:
        _last_source[backend.key] = "path"
        return default_path

    logger.info("backend %s resolved from mlflow registry: %s", backend.key, resolved)
    _last_source[backend.key] = "registry"
    return resolved


def _resolve_from_registry(backend: Backend, settings: Settings) -> Path | None:
    """Look up the configured stage's version of ``settings.mlflow_model_name``.

    Returns ``None`` (never raises) for "nothing to promote" outcomes -- no
    Production version, or one logged for a different backend -- so the
    caller falls back exactly like it would on a network failure.
    """
    # Lazy on purpose: this is the only place `mlflow` is imported, and only
    # once a tracking URI is configured, so the default path never pays for a
    # server client it does not use.
    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", _REQUEST_TIMEOUT_S)

    # The client's own ``tracking_uri`` is not enough. A server started with
    # ``--serve-artifacts`` hands out ``mlflow-artifacts:/`` URIs, and those are
    # resolved against the *global* tracking URI, not the client's -- so
    # downloading would look in the default local ``mlruns`` and fail with
    # "the tracking URI must be a valid http or https URI". Setting it globally
    # is safe here because this process talks to exactly one tracking server.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_name = settings.mlflow_model_name
    if not model_name:
        logger.warning("mlflow_tracking_uri is set but mlflow_model_name is empty")
        return None
    stage = settings.mlflow_stage or "Production"

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        logger.warning("no %s version of %s in the registry", stage, model_name)
        return None
    version = versions[0]

    run = client.get_run(version.run_id)
    logged_backend = run.data.params.get("backend")
    if logged_backend != backend.key:
        logger.warning(
            "%s v%s was logged for backend %s, not %s; format mismatch, ignoring",
            model_name,
            version.version,
            logged_backend,
            backend.key,
        )
        return None

    return _cached_download(client, version, backend, settings)


# mlflow's `ModelVersion` / `MlflowClient` types only exist once the lazy
# import above has run, so they cannot be named in a module-level annotation.
# `Any` is deliberate here, the same trade-off `ANN401` is already ignored
# for at the Ultralytics boundary in `detector.py`.
def _cached_download(client: Any, version: Any, backend: Backend, settings: Settings) -> Path:
    """Download the version's artifact once, then reuse it on every later call.

    Cached per (model, version, backend) under ``models_dir``, so a hot-swap
    that lands back on an already-resolved backend costs a marker-file check,
    not a re-download -- re-fetching on every switch would make hot-swap
    unusable.
    """
    cache_root = (
        settings.models_dir
        / "_mlflow_cache"
        / settings.mlflow_model_name
        / str(version.version)
        / backend.key
    )
    marker = cache_root / ".complete"
    cached = _artifact_within(cache_root / "model", backend)
    # The marker alone is not proof: a pruned or half-deleted cache leaves it
    # behind, and returning a path to a missing file would fail the load and
    # blacklist the backend for the process -- a non-OOM failure is treated as
    # permanent. Checking the artifact itself makes a damaged cache self-heal.
    if marker.exists() and cached.exists():
        return cached

    cache_root.mkdir(parents=True, exist_ok=True)
    # `download_artifacts` places files under `dst_path/<artifact_path>`,
    # i.e. `cache_root/model`, and returns that path -- captured rather
    # than assumed, in case a future mlflow version changes the layout.
    downloaded_dir = Path(client.download_artifacts(version.run_id, "model", str(cache_root)))
    marker.touch()
    return _artifact_within(downloaded_dir, backend)


def _artifact_within(downloaded_dir: Path, backend: Backend) -> Path:
    """The downloaded artifact: the directory itself, or the one file in it.

    OpenVINO IR is logged as a directory (``mlflow.log_artifacts``), so the
    download *is* the artifact. Every other backend is logged as a single
    file (``mlflow.log_artifact``) with nothing else alongside it, so the
    file with its original name -- not the folder holding it -- is what the
    caller wants.
    """
    suffix = Path(backend.artifact).suffix
    if not suffix:
        return downloaded_dir
    return downloaded_dir / Path(backend.artifact).name


def _short(exc: BaseException, limit: int = 200) -> str:
    """One-line exception text; registry errors can be paragraphs of HTTP detail."""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else f"{text[:limit]}..."
