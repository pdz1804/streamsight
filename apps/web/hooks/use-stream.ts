"use client";

/**
 * WebSocket streaming hook.
 *
 * Three deliberate performance decisions:
 *
 * 1. **Frames never enter React state.** Each annotated JPEG is painted straight
 *    to a canvas as it arrives. Putting a ~100 KB payload into state 30 times a
 *    second would re-render the tree on every frame for no benefit.
 * 2. **Painting is immediate, telemetry is throttled.** The canvas draw happens
 *    the instant a frame is parsed, with no `requestAnimationFrame` pacing --
 *    rAF is throttled or suspended in background and unfocused tabs, which
 *    stalls the visible stream exactly when someone is watching the network
 *    panel. Only the numeric readouts are batched, at 4 Hz, because no human
 *    reads a latency figure 30 times a second.
 * 3. **Pixels arrive as bytes, not base64.** The server sends one binary message
 *    per frame -- a JSON header followed by the raw JPEG -- and those bytes go
 *    to `createImageBitmap`, which decodes off the main thread. The older
 *    `data:` URI path fed `new Image()`, which decodes on the main thread and
 *    carries a third more bytes to get there. It remains as a fallback for
 *    browsers without `createImageBitmap`.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { streamUrl } from "@/lib/api";
import type { StreamFrame, StreamMessage, StreamPhase, Track } from "@/lib/types";

const TELEMETRY_INTERVAL_MS = 250;

/** Width of the big-endian uint32 length prefix on every binary frame. */
const HEADER_LENGTH_BYTES = 4;

const headerDecoder = new TextDecoder();

export interface StreamTelemetry {
  frameId: number;
  fps: number;
  serverLatencyMs: number;
  inferenceMs: number;
  clientLatencyMs: number;
  precision: string;
  imgsz: number;
  degradedMode: boolean;
  tracks: Track[];
  width: number;
  height: number;
}

const EMPTY_TELEMETRY: StreamTelemetry = {
  frameId: 0,
  fps: 0,
  serverLatencyMs: 0,
  inferenceMs: 0,
  clientLatencyMs: 0,
  precision: "",
  imgsz: 0,
  degradedMode: false,
  tracks: [],
  width: 0,
  height: 0,
};

export interface UseStreamResult {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  phase: StreamPhase | "idle";
  message: string;
  telemetry: StreamTelemetry;
  connected: boolean;
  start: (source: string, loop?: boolean) => void;
  stop: () => void;
}

export function useStream(): UseStreamResult {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const lastPaintedRef = useRef(0);
  const pendingRef = useRef<StreamTelemetry | null>(null);
  const flushTimerRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);

  const [phase, setPhase] = useState<StreamPhase | "idle">("idle");
  const [message, setMessage] = useState("");
  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState<StreamTelemetry>(EMPTY_TELEMETRY);

  /** Batch numeric updates so the side panel repaints 4x/s, not 30x/s. */
  const queueTelemetry = useCallback((next: StreamTelemetry) => {
    pendingRef.current = next;
    if (flushTimerRef.current !== null) return;
    flushTimerRef.current = window.setTimeout(() => {
      flushTimerRef.current = null;
      if (pendingRef.current) setTelemetry(pendingRef.current);
    }, TELEMETRY_INTERVAL_MS);
  }, []);

  /**
   * Draw a decoded frame, unless a newer one has already been painted.
   *
   * Decoding is async, so a slow frame can land after a newer one; dropping it
   * keeps the stream monotonic instead of flickering backwards.
   */
  const drawSource = useCallback(
    (frameId: number, source: CanvasImageSource, width: number, height: number) => {
      const canvas = canvasRef.current;
      if (!canvas || frameId < lastPaintedRef.current) return;
      lastPaintedRef.current = frameId;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      canvas.getContext("2d")?.drawImage(source, 0, 0);
    },
    [],
  );

  /** Decode JPEG bytes off the main thread and paint them. */
  const paintBytes = useCallback(
    (frameId: number, jpeg: Uint8Array<ArrayBuffer>) => {
      const blob = new Blob([jpeg], { type: "image/jpeg" });
      if (typeof createImageBitmap === "function") {
        void createImageBitmap(blob)
          .then((bitmap) => {
            drawSource(frameId, bitmap, bitmap.width, bitmap.height);
            // ImageBitmaps hold their pixel buffer until closed explicitly;
            // leaving that to the GC leaks tens of MB within a minute.
            bitmap.close();
          })
          .catch(() => undefined);
        return;
      }
      // No createImageBitmap: decode via an object URL, which still beats a
      // data URI because the bytes are never stringified.
      const url = URL.createObjectURL(blob);
      const image = new Image();
      image.onload = () => {
        drawSource(frameId, image, image.width, image.height);
        URL.revokeObjectURL(url);
      };
      image.onerror = () => URL.revokeObjectURL(url);
      image.src = url;
    },
    [drawSource],
  );

  /** Split a binary message into its JSON header and the JPEG that follows. */
  const readBinaryFrame = useCallback((buffer: ArrayBuffer): StreamFrame | null => {
    if (buffer.byteLength < HEADER_LENGTH_BYTES) return null;
    const headerLength = new DataView(buffer).getUint32(0, false);
    const jpegStart = HEADER_LENGTH_BYTES + headerLength;
    if (buffer.byteLength < jpegStart) return null;
    const header = JSON.parse(
      headerDecoder.decode(new Uint8Array(buffer, HEADER_LENGTH_BYTES, headerLength)),
    ) as StreamFrame;
    paintBytes(header.frame_id, new Uint8Array(buffer, jpegStart));
    return header;
  }, [paintBytes]);

  /** Legacy transport: the pixels arrive as a base64 data URI inside the JSON. */
  const paintDataUri = useCallback(
    (frame: StreamFrame) => {
      if (!frame.image) return;
      const image = new Image();
      image.onload = () => drawSource(frame.frame_id, image, image.width, image.height);
      image.src = frame.image;
    },
    [drawSource],
  );

  const stop = useCallback(() => {
    intentionalCloseRef.current = true;
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: "stop" }));
    }
    socket?.close();
    socketRef.current = null;
    setConnected(false);
    setPhase("idle");
  }, []);

  const start = useCallback(
    (source: string, loop = true) => {
      socketRef.current?.close();
      intentionalCloseRef.current = false;
      lastPaintedRef.current = 0;
      setPhase("opening");
      setMessage("connecting");

      const socket = new WebSocket(streamUrl(source, loop));
      // Frames arrive as ArrayBuffers rather than Blobs so the header can be
      // read synchronously; a Blob would force an extra async hop per frame.
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => setConnected(true);

      socket.onmessage = (event) => {
        let payload: StreamMessage | null;
        if (event.data instanceof ArrayBuffer) {
          payload = readBinaryFrame(event.data);
        } else {
          payload = JSON.parse(event.data as string) as StreamMessage;
          if (payload.kind === "status") {
            setPhase(payload.phase);
            setMessage(payload.message);
            return;
          }
          paintDataUri(payload);
        }
        if (!payload || payload.kind !== "frame") return;
        queueTelemetry({
          frameId: payload.frame_id,
          fps: payload.fps,
          serverLatencyMs: payload.timing.total_ms,
          inferenceMs: payload.timing.inference_ms,
          clientLatencyMs: Math.max(0, Date.now() - payload.server_ts),
          precision: payload.precision,
          imgsz: payload.imgsz,
          degradedMode: payload.degraded_mode,
          tracks: payload.tracks,
          width: payload.width,
          height: payload.height,
        });
      };

      socket.onerror = () => {
        setPhase("error");
        setMessage("stream connection failed");
      };

      socket.onclose = () => {
        setConnected(false);
        if (!intentionalCloseRef.current) {
          setPhase((current) => (current === "error" ? current : "ended"));
        }
      };
    },
    [paintDataUri, readBinaryFrame, queueTelemetry],
  );

  useEffect(
    () => () => {
      intentionalCloseRef.current = true;
      socketRef.current?.close();
      if (flushTimerRef.current !== null) window.clearTimeout(flushTimerRef.current);
    },
    [],
  );

  return { canvasRef, phase, message, telemetry, connected, start, stop };
}
