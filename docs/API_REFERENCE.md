# API reference

Base URL `http://localhost:8100`. Interactive docs at `/docs`.

Domain errors return `{"error": "<ExceptionName>", "detail": "<message>"}` with a meaningful status:
`400` undecodable input, `409` backend cannot serve the request, `503` no backend is loaded.

## `GET /health`

Liveness plus the active configuration. Answers even before a model is ready, so a probe can tell
"starting" from "broken".

```json
{
  "status": "ok",
  "app": "StreamSight",
  "version": "0.1.0",
  "gpu": { "available": true, "name": "NVIDIA RTX A1000 Laptop GPU",
           "total_mb": 4096, "used_mb": 436, "free_mb": 3659 },
  "precision": "fp32_gpu",
  "imgsz": 640
}
```

## `POST /detect/frame`

Detect and track one base64 image. Accepts a bare base64 string or a `data:image/...;base64,` URI.

```json
{ "image": "data:image/jpeg;base64,/9j/4AAQ..." }
```

Response (abridged):

```json
{
  "frame_id": 42,
  "width": 1920, "height": 1080,
  "detections": [
    { "x1": 812.4, "y1": 301.7, "x2": 954.0, "y2": 688.2,
      "confidence": 0.91, "class_id": 0, "class_name": "person" }
  ],
  "tracks": [
    { "x1": 812.4, "y1": 301.7, "x2": 954.0, "y2": 688.2,
      "confidence": 0.91, "class_id": 0, "class_name": "person", "track_id": 1 }
  ],
  "timing": { "decode_ms": 0.0, "inference_ms": 96.3, "encode_ms": 0.0, "total_ms": 96.8 },
  "latency_ms": 96.8,
  "fps": 13.4, "precision": "fp32_gpu", "imgsz": 640, "degraded_mode": false
}
```

`latency_ms` is a flat mirror of `timing.total_ms`, derived from it rather than stored separately so
the two cannot disagree. Use whichever suits the caller; `timing` keeps the per-stage breakdown.

`detections` holds every box in the frame. `tracks` holds only those ByteTrack has assigned a
persistent id. A box without an id is still detected, it just has not survived enough frames to be
confirmed, so it is correctly absent from `tracks` rather than given an invented id.

## `POST /detect/image`

Same as above with a multipart file upload (`file=@frame.jpg`).

## `WS /detect/stream`

Query: `source` (source id, device index, or `rtsp://` URL), `loop` (restart file sources),
`encoding` (`binary` — the default — or `base64`).

**Status messages are always JSON text**, in both encodings.

```json
{ "kind": "status", "phase": "streaming",
  "message": "file 1280x720 @ 30 fps", "source": "...", "total_frames": 984 }
```

`phase` is one of `opening`, `streaming`, `ended`, `error`.

### Frames — `encoding=binary` (default)

Each frame is **one binary message** laid out as:

```
[4-byte big-endian uint32 header length][UTF-8 JSON header][raw JPEG bytes]
```

The header is the frame object below with `image` omitted. Pixels and metadata travel in a single
message on purpose: as two messages they could interleave under concurrent sends and pair a header
with the wrong image, so a client would never have to correlate them — and cannot get it wrong.

Clients discriminate on the message type alone — binary means frame, text means status. Sniffing
content is never required.

The `image` field is **absent** from a binary header — not `null`. The pixels are the bytes that
follow it.

```jsonc
{ "kind": "frame", "frame_id": 117,
  "width": 1280, "height": 720,
  "tracks": [ /* ... */ ],
  "timing": { "inference_ms": 31.1, "encode_ms": 12.4, "total_ms": 43.5,
              "wait_ms": 0.1, "send_ms": 2.8 },
  "fps": 13.4, "server_ts": 1784500000000.0,
  "precision": "fp32_gpu", "imgsz": 640, "degraded_mode": false }
```

`send_ms` reports the **previous** frame's send cost. A frame is already serialized by the time its
own send finishes, so it can never carry that figure itself.

### Frames — `encoding=base64`

The original transport, kept for clients that cannot read binary messages. One JSON text message per
frame, with `image` holding a `data:image/jpeg;base64,...` URI and nothing appended.

It is not the default because base64 costs a third more bytes plus an encode on the server and a
decode in the client, and a `data:` URI decodes on the browser's main thread.

### Both encodings

The overlay is already burned into the image; `tracks` is for the legend and inspection. `server_ts`
is a send timestamp in epoch milliseconds, which the client subtracts to display end-to-end latency.

Frames are produced and sent by separate tasks with a depth-1 queue between them, and the queue
**drops the oldest** pending frame under backpressure. A slow client therefore sees gaps in
`frame_id`, never a growing delay — a live view must stay current rather than complete.

Send `{"action":"stop"}` to end the stream; closing the socket works too.

## `GET /metrics`

Rolling telemetry: `fps`, `fps_rolling`, `avg/p50/p95_latency_ms`, `frames_processed`,
`track_count`, `unique_tracks`, `gpu`, `cpu_percent`, `ram_used_mb`, `process_ram_mb`,
`degraded_mode`, `degrade_reason`, `precision`, `imgsz`, `uptime_s`.

`gpu_mem_mb` is also served as a flat field, mirroring `gpu.used_mb` for single-gauge dashboards.
Like `latency_ms` it is derived, not stored: the nested `gpu` block stays the source of truth.

FPS is derived from frame arrival times rather than `1000 / latency`, so queueing and encode cost
are included. It is the rate a viewer actually perceives.

## `GET /config/model`

Active configuration plus the full selectable menu. Every backend is listed whether or not it can
run; unavailable ones always carry a `reason`.

```json
{
  "precision": "fp32_gpu", "imgsz": 640, "device": "cuda",
  "model_file": "yolo11n.pt", "degraded_mode": false, "degrade_reason": null,
  "available_backends": [
    { "precision": "int8_trt", "label": "INT8 - TensorRT", "device": "cuda",
      "available": false, "reason": "artifact missing (run ml/quantization exports)",
      "artifact": "engines/yolo11n_int8.engine" }
  ],
  "supported_imgsz": [640, 480, 320]
}
```

## `POST /config/model`

Hot-swap precision and/or resolution without restarting. Omitted fields keep their current value;
unknown fields are rejected (`422`) so a typo cannot silently do nothing.

```json
{ "precision": "openvino_cpu", "imgsz": 640 }
```

**Resolution.** `resolution` is accepted as a synonym for `imgsz`. Sending both is fine when they
agree; sending different values is a contradiction and returns `422` rather than a guess.

**Precision vocabulary.** `precision` takes either a concrete backend key (`int8_trt`, `fp16_trt`,
`fp16_onnx`, `fp32_gpu`, `openvino_cpu`, `int8_onnx_cpu`, `fp32_cpu`) or one of the abstract words
`int8` | `fp16` | `fp32`. An abstract word resolves to the first backend on its list that is
actually runnable on this host at the requested resolution:

| Word | Tried in order |
|---|---|
| `int8` | `int8_trt` → `int8_onnx_cpu` |
| `fp16` | `fp16_trt` → `fp16_onnx` |
| `fp32` | `fp32_gpu` → `fp32_cpu` |

So `int8` lands on TensorRT where it exists and on the ONNX CPU graph where it does not. If no
candidate can run, the `409` names every one that was tried and why — for example, `fp16` on a
machine with no NVIDIA GPU, since both FP16 artifacts are CUDA-only.

Track identities reset across a swap: ids from the previous model are not comparable to the new
one's. If the requested backend cannot be loaded, the previously working configuration is restored
before the `409` is returned, so a failed request never leaves the service dead.

## `POST /config/degrade`

Forces one step down the degradation ladder, exactly as a CUDA out-of-memory event would. Returns
the new configuration. Present so the fallback path can be demonstrated rather than assumed.

## `GET /sources` · `POST /sources/upload`

Lists selectable sources (bundled clip, webcam, uploads) and accepts a video upload. Uploads are
stored under generated ids; the client never influences a filesystem path. Extension and size are
validated, and rejects explain the allowed types.

Phu Nguyen - HCMC, Vietnam
