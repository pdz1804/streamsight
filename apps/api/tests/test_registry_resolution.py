"""MLflow-backed artifact resolution: default path, fallback, and caching.

Every mlflow-touching test fakes `mlflow.tracking.MlflowClient` via
`sys.modules` rather than hitting a real server -- the resolver imports
mlflow lazily specifically so this works without the package (or a running
server) being involved at all. The one test that must prove "unset config
means zero import" goes further and blocks the import outright.
"""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from app.core.config import Settings
from app.inference.backends import BACKENDS
from app.inference.registry import (
    last_resolution_source,
    reset_registry_state,
    resolve_artifact,
    resolved_backend,
)

MODEL_NAME = "streamsight-detector"
TRACKING_URI = "http://127.0.0.1:5000"


@pytest.fixture(autouse=True)
def _clean_registry_state() -> None:
    """The resolver remembers an unreachable registry process-wide; tests must not."""
    reset_registry_state()


@dataclass
class FakeVersion:
    version: str
    run_id: str


@dataclass
class FakeRunData:
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeRun:
    data: FakeRunData


def _make_fake_client_class(
    *,
    versions: dict[str, list[FakeVersion]] | None = None,
    runs: dict[str, FakeRun] | None = None,
    unreachable: bool = False,
    is_directory: bool = False,
    artifact_filename: str = "yolo11n_int8.onnx",
) -> type:
    """Build a fresh fake `MlflowClient` class bound to this test's fixtures.

    A fresh class per call (not a shared one) so `download_calls` -- tracked
    on the class because the resolver constructs a new client per call, like
    the real one -- never leaks between tests.
    """
    versions = versions or {}
    runs = runs or {}

    class _FakeMlflowClient:
        download_calls = 0

        def __init__(self, tracking_uri: str | None = None) -> None:
            self.tracking_uri = tracking_uri

        lookup_calls = 0

        def get_latest_versions(self, name: str, stages: list[str]) -> list[FakeVersion]:
            _FakeMlflowClient.lookup_calls += 1
            if unreachable:
                raise ConnectionError("mlflow server unreachable")
            return versions.get(name, [])

        def get_run(self, run_id: str) -> FakeRun:
            return runs[run_id]

        def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
            _FakeMlflowClient.download_calls += 1
            target_dir = Path(dst_path) / path
            target_dir.mkdir(parents=True, exist_ok=True)
            if is_directory:
                (target_dir / "model.xml").write_text("<net/>", encoding="utf-8")
                (target_dir / "model.bin").write_bytes(b"fake-openvino-weights")
            else:
                (target_dir / artifact_filename).write_bytes(b"fake-weights")
            return str(target_dir)

    return _FakeMlflowClient


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch, client_cls: type) -> list[str]:
    """Install a fake ``mlflow``; returns the list that records tracking-URI sets.

    ``set_tracking_uri`` is part of the surface under test, not incidental. A
    server started with ``--serve-artifacts`` returns ``mlflow-artifacts:/``
    URIs, which resolve against the *global* tracking URI rather than the
    client's -- so without that call artifact downloads go looking in the local
    default store and fail. A fake that omitted it let that bug through.
    """
    tracking_uris: list[str] = []
    tracking_module = types.ModuleType("mlflow.tracking")
    tracking_module.MlflowClient = client_cls
    mlflow_module = types.ModuleType("mlflow")
    mlflow_module.tracking = tracking_module
    mlflow_module.set_tracking_uri = tracking_uris.append
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_module)
    return tracking_uris


class _BlockMlflowImport:
    """A meta path finder that turns any `import mlflow*` into an ImportError."""

    def find_spec(self, name: str, path: object = None, target: object = None) -> None:
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("mlflow must not be imported when mlflow_tracking_uri is unset")
        return None


# --------------------------------------------------------------------------- #
# default path: unconfigured means unchanged, and no mlflow import at all
# --------------------------------------------------------------------------- #


def test_unset_tracking_uri_returns_default_path_without_importing_mlflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    already_imported = [n for n in list(sys.modules) if n == "mlflow" or n.startswith("mlflow.")]
    for name in already_imported:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockMlflowImport(), *sys.meta_path])

    settings = Settings(models_dir=tmp_path / "models")
    backend = BACKENDS["int8_onnx_cpu"]

    result = resolve_artifact(backend, settings)

    assert result == backend.path(settings)
    assert last_resolution_source(backend.key) == "path"


def test_resolved_backend_is_the_same_object_when_unconfigured(tmp_path: Path) -> None:
    settings = Settings(models_dir=tmp_path / "models")
    backend = BACKENDS["int8_onnx_cpu"]

    assert resolved_backend(backend, settings) is backend


# --------------------------------------------------------------------------- #
# registry unreachable: fall back, but loudly
# --------------------------------------------------------------------------- #


def test_registry_unreachable_falls_back_to_path_and_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_mlflow(monkeypatch, _make_fake_client_class(unreachable=True))
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name=MODEL_NAME,
    )
    backend = BACKENDS["int8_onnx_cpu"]

    with caplog.at_level(logging.WARNING, logger="app.inference.registry"):
        result = resolve_artifact(backend, settings)

    assert result == backend.path(settings)
    assert last_resolution_source(backend.key) == "path"
    assert any("mlflow unreachable" in r.message for r in caplog.records)


def test_an_unreachable_registry_is_contacted_once_not_once_per_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Startup walks a fallback ladder; a dead server must not be probed at every rung.

    Each probe costs the full HTTP timeout, so retrying per backend would turn
    one unreachable server into a boot delay measured in tens of seconds --
    precisely the startup coupling this resolver is built to avoid.
    """
    client_cls = _make_fake_client_class(unreachable=True)
    _install_fake_mlflow(monkeypatch, client_cls)
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name=MODEL_NAME,
    )

    for backend in BACKENDS.values():
        assert resolve_artifact(backend, settings) == backend.path(settings)

    assert len(BACKENDS) > 1
    assert client_cls.lookup_calls == 1


def test_a_configuration_error_does_not_silence_the_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only transport failures earn the cooldown.

    A bad model name or a deleted run fails identically every time and costs
    nothing to re-ask. Treating it like an unreachable server would disable a
    perfectly healthy registry for every other backend and every later
    hot-swap, on the strength of one misconfigured field.
    """

    class _ExplodingClient:
        lookup_calls = 0

        def __init__(self, tracking_uri: str | None = None) -> None:
            self.tracking_uri = tracking_uri

        def get_latest_versions(self, name: str, stages: list[str]) -> list[FakeVersion]:
            _ExplodingClient.lookup_calls += 1
            raise ValueError(f"RESOURCE_DOES_NOT_EXIST: no registered model named {name}")

    _install_fake_mlflow(monkeypatch, _ExplodingClient)
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name="typo-in-the-model-name",
    )

    for backend in list(BACKENDS.values())[:3]:
        assert resolve_artifact(backend, settings) == backend.path(settings)

    assert _ExplodingClient.lookup_calls == 3


def test_format_mismatch_falls_back_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Production version logged for a different backend must be ignored."""
    version = FakeVersion(version="1", run_id="run-fp16")
    run = FakeRun(data=FakeRunData(params={"backend": "fp16_onnx"}))
    client_cls = _make_fake_client_class(versions={MODEL_NAME: [version]}, runs={"run-fp16": run})
    _install_fake_mlflow(monkeypatch, client_cls)
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name=MODEL_NAME,
    )
    backend = BACKENDS["int8_onnx_cpu"]

    result = resolve_artifact(backend, settings)

    assert result == backend.path(settings)
    assert client_cls.download_calls == 0


# --------------------------------------------------------------------------- #
# production hit: cached artifact, downloaded exactly once
# --------------------------------------------------------------------------- #


def test_production_hit_returns_a_downloaded_cached_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    version = FakeVersion(version="3", run_id="run-int8")
    run = FakeRun(data=FakeRunData(params={"backend": "int8_onnx_cpu"}))
    client_cls = _make_fake_client_class(
        versions={MODEL_NAME: [version]},
        runs={"run-int8": run},
        artifact_filename="yolo11n_int8.onnx",
    )
    tracking_uris = _install_fake_mlflow(monkeypatch, client_cls)
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name=MODEL_NAME,
    )
    backend = BACKENDS["int8_onnx_cpu"]

    result = resolve_artifact(backend, settings)

    assert result != backend.path(settings)
    assert result.name == "yolo11n_int8.onnx"
    assert result.exists()
    assert last_resolution_source(backend.key) == "registry"
    assert client_cls.download_calls == 1
    # Without this the download resolves `mlflow-artifacts:/` against the local
    # default store and fails against any --serve-artifacts server.
    assert tracking_uris == [TRACKING_URI]


def test_second_resolution_reuses_the_cached_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    version = FakeVersion(version="3", run_id="run-int8")
    run = FakeRun(data=FakeRunData(params={"backend": "int8_onnx_cpu"}))
    client_cls = _make_fake_client_class(versions={MODEL_NAME: [version]}, runs={"run-int8": run})
    _install_fake_mlflow(monkeypatch, client_cls)
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name=MODEL_NAME,
    )
    backend = BACKENDS["int8_onnx_cpu"]

    first = resolve_artifact(backend, settings)
    second = resolve_artifact(backend, settings)

    assert first == second
    assert client_cls.download_calls == 1


def test_directory_backend_resolves_to_the_downloaded_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OpenVINO IR is a directory artifact; resolution must not pick one file."""
    version = FakeVersion(version="1", run_id="run-ov")
    run = FakeRun(data=FakeRunData(params={"backend": "openvino_cpu"}))
    client_cls = _make_fake_client_class(
        versions={MODEL_NAME: [version]}, runs={"run-ov": run}, is_directory=True
    )
    _install_fake_mlflow(monkeypatch, client_cls)
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name=MODEL_NAME,
    )
    backend = BACKENDS["openvino_cpu"]

    result = resolve_artifact(backend, settings)

    assert result.is_dir()
    assert (result / "model.xml").exists()


def test_resolved_backend_overrides_artifact_on_registry_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    version = FakeVersion(version="2", run_id="run-hot-swap")
    run = FakeRun(data=FakeRunData(params={"backend": "int8_onnx_cpu"}))
    client_cls = _make_fake_client_class(
        versions={MODEL_NAME: [version]}, runs={"run-hot-swap": run}
    )
    _install_fake_mlflow(monkeypatch, client_cls)
    settings = Settings(
        models_dir=tmp_path / "models",
        mlflow_tracking_uri=TRACKING_URI,
        mlflow_model_name=MODEL_NAME,
    )
    backend = BACKENDS["int8_onnx_cpu"]

    effective = resolved_backend(backend, settings)

    assert effective.key == backend.key
    assert effective.path(settings) != backend.path(settings)
    assert effective.path(settings).exists()
