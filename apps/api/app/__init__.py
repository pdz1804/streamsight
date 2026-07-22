"""StreamSight API package.

Importing this package disables Ultralytics' auto-install behaviour. Left on, it
runs ``pip install`` mid-inference when it thinks an export dependency is
missing, which on this project silently installed a CPU ``onnxruntime`` over the
GPU build and left an environment where even ``import torch`` failed. Dependency
changes belong in requirements files, never in a request handler.
"""

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

__version__ = "0.1.0"
