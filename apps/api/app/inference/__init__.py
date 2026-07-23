"""Model ownership: which backend is loaded, how it is chosen, and how it runs.

:mod:`app.inference.runtime` is the single mutator of model state -- startup
selection, hot-swap, and automatic degradation all funnel through it under one
lock. Nothing outside this package constructs a
:class:`~app.inference.detector.Detector` directly.
"""
