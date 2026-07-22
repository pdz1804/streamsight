"use client";

/**
 * WebSocket streaming hook.
 *
 * Two deliberate performance decisions:
 *
 * 1. **Frames never enter React state.** Each annotated JPEG is painted straight
 *    to a canvas as it arrives. Putting a ~100 KB data URI into state 30 times a
 *    second would re-render the tree on every frame for no benefit.
 * 2. **Painting is immediate, telemetry is throttled.** The canvas draw happens
 *    the instant a frame is parsed, with no `requestAnimationFrame` pacing --
 *    rAF is throttled or suspended in background and unfocused tabs, which
 *    stalls the visible stream exactly when someone is watching the network
 *    panel. Only the numeric readouts are batched, at 4 Hz, because no human
 *    reads a latency figure 30 times a second.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { streamUrl } from "./api";
import type { StreamFrame, StreamMessage, StreamPhase, Track } from "./types";

const TELEMETRY_INTERVAL_MS = 250;

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

  const paint = useCallback((frame: StreamFrame) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const image = new Image();
    image.onload = () => {
      // Decoding is async, so a slow frame could land after a newer one.
      // Dropping it keeps the stream monotonic instead of flickering backwards.
      if (frame.frame_id < lastPaintedRef.current) return;
      lastPaintedRef.current = frame.frame_id;
      if (canvas.width !== image.width || canvas.height !== image.height) {
        canvas.width = image.width;
        canvas.height = image.height;
      }
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(image, 0, 0);
    };
    image.src = frame.image;
  }, []);

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
      socketRef.current = socket;

      socket.onopen = () => setConnected(true);

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data as string) as StreamMessage;
        if (payload.kind === "status") {
          setPhase(payload.phase);
          setMessage(payload.message);
          return;
        }
        paint(payload);
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
    [paint, queueTelemetry],
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
