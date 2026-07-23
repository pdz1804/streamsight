"""FastAPI dependency providers.

The runtime and source registry are built once in the lifespan handler and stored
on ``app.state``; these accessors are the only sanctioned way to reach them, which
keeps routers free of module-level globals and trivially overridable in tests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, WebSocket

from .core.config import Settings, get_settings
from .core.exceptions import NoBackendError
from .inference.runtime import InferenceRuntime
from .streaming.sources import SourceRegistry


def get_runtime(request: Request) -> InferenceRuntime:
    runtime: InferenceRuntime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.ready:
        raise NoBackendError("inference runtime is not ready")
    return runtime


def get_runtime_ws(websocket: WebSocket) -> InferenceRuntime:
    runtime: InferenceRuntime | None = getattr(websocket.app.state, "runtime", None)
    if runtime is None or not runtime.ready:
        raise NoBackendError("inference runtime is not ready")
    return runtime


def get_registry(request: Request) -> SourceRegistry:
    return request.app.state.registry


def get_registry_ws(websocket: WebSocket) -> SourceRegistry:
    return websocket.app.state.registry


RuntimeDep = Annotated[InferenceRuntime, Depends(get_runtime)]
RuntimeWsDep = Annotated[InferenceRuntime, Depends(get_runtime_ws)]
RegistryDep = Annotated[SourceRegistry, Depends(get_registry)]
RegistryWsDep = Annotated[SourceRegistry, Depends(get_registry_ws)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
