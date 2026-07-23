"""Observability: rolling latency/throughput windows and the durable track log.

Read by ``GET /metrics`` and written from the inference hot path, so everything
here has to stay cheap enough to call once per frame.
"""
