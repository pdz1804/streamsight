"""Lightweight stand-ins for Ultralytics result objects.

Parsing logic should be testable without loading a 5 MB model and running real
inference, so these mimic only the surface `parse_results` touches: a `boxes`
attribute exposing torch-like tensors, and a `names` mapping.

Kept in its own module rather than in `conftest.py` because an unrelated
top-level `tests` package exists in site-packages, and importing
`tests.conftest` resolves to that one instead of ours.
"""

from __future__ import annotations

import numpy as np


class _Tensor:
    """Mimics the torch tensor surface used by result parsing."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __len__(self) -> int:
        return len(self._array)


class FakeBoxes:
    """Minimal stand-in for ``ultralytics.engine.results.Boxes``."""

    def __init__(self, xyxy, conf, cls, ids=None) -> None:
        self.xyxy = _Tensor(np.asarray(xyxy, dtype=np.float32))
        self.conf = _Tensor(np.asarray(conf, dtype=np.float32))
        self.cls = _Tensor(np.asarray(cls, dtype=np.float32))
        self.id = _Tensor(np.asarray(ids, dtype=np.float32)) if ids is not None else None

    def __len__(self) -> int:
        return len(self.xyxy.numpy())


class FakeResults:
    """Minimal stand-in for ``ultralytics.engine.results.Results``."""

    def __init__(self, boxes: FakeBoxes | None, names: dict[int, str] | None = None) -> None:
        self.boxes = boxes
        self.names = names or {0: "person", 2: "car"}
