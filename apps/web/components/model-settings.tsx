"use client";

/**
 * Model selector and reliability drill.
 *
 * Unavailable backends stay visible with the reason they cannot run, rather than
 * being hidden. On a 4 GB laptop "why can I not pick INT8 TensorRT" is the first
 * question anyone asks, and the answer is almost always a missing export step.
 */

import { ArrowsClockwise, Warning } from "@phosphor-icons/react/dist/ssr";
import { useCallback, useEffect, useState } from "react";

import { api, ApiError } from "@/lib/api";
import type { ModelConfigResponse } from "@/lib/types";

import { Button, Chip, ErrorNote, Field, Panel, Skeleton } from "./ui";

/** Backend failures quote raw driver errors, which run to several lines. */
function truncate(text: string, limit = 84): string {
  return text.length <= limit ? text : `${text.slice(0, limit).trimEnd()}...`;
}

export function ModelSettings() {
  const [config, setConfig] = useState<ModelConfigResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setConfig(await api.modelConfig());
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const apply = async (payload: { precision?: string; imgsz?: number }) => {
    setBusy(true);
    setError(null);
    try {
      setConfig(await api.setModelConfig(payload));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
      await load();
    } finally {
      setBusy(false);
    }
  };

  const drill = async () => {
    setBusy(true);
    setError(null);
    try {
      setConfig(await api.forceDegrade());
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  if (!config) {
    return (
      <div className="grid gap-4 p-4 md:p-6 lg:grid-cols-2">
        <Skeleton className="h-80" />
        <Skeleton className="h-48" />
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-[1500px] flex-col gap-5 p-5 md:p-7">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-[22px] font-semibold tracking-tight text-text">Inference configuration</h1>
        {config.degraded_mode ? <Chip tone="warn">Degraded</Chip> : <Chip tone="ok">Nominal</Chip>}
        <span className="font-mono text-[11px] text-text-mute">
          {config.model_file} on {config.device}
        </span>
        <Button className="ml-auto" onClick={() => void load()} disabled={busy}>
          <ArrowsClockwise size={15} />
          Refresh
        </Button>
      </div>

      {error ? <ErrorNote>{error}</ErrorNote> : null}
      {config.degrade_reason ? (
        <p className="rounded-[var(--radius)] border border-warn/40 bg-warn/8 px-3 py-2 text-[12px] text-warn">
          {config.degrade_reason}
        </p>
      ) : null}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Panel title="Backend" index={0}>
          <ul className="flex flex-col gap-1 px-2 pb-4">
            {config.available_backends.map((backend) => {
              const active = backend.precision === config.precision;
              return (
                <li key={backend.precision}>
                  <button
                    type="button"
                    disabled={!backend.available || busy || active}
                    onClick={() => void apply({ precision: backend.precision })}
                    className={`flex w-full items-start gap-3 rounded-[var(--radius)] border px-3.5 py-3 text-left transition-all duration-200 ${
                      active
                        ? "border-accent/40 bg-accent-soft shadow-[var(--shadow-sm)]"
                        : "border-transparent hover:border-line hover:bg-surface-2"
                    } disabled:cursor-not-allowed disabled:hover:border-transparent disabled:hover:bg-transparent`}
                  >
                    <span
                      aria-hidden="true"
                      className={`mt-1 size-3.5 shrink-0 rounded-full border-2 ${
                        active ? "border-accent bg-accent" : "border-line-strong"
                      }`}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex flex-wrap items-center gap-2">
                        <span
                          className={`text-[13px] font-medium ${
                            backend.available ? "text-text" : "text-text-mute"
                          }`}
                        >
                          {backend.label}
                        </span>
                        <span className="font-mono text-[10px] uppercase tracking-wider text-text-mute">
                          {backend.device}
                        </span>
                        {!backend.available ? (
                          <span title={backend.reason}>
                            <Chip tone="neutral">{truncate(backend.reason)}</Chip>
                          </span>
                        ) : null}
                      </span>
                      <span className="mt-1 block text-[12px] leading-relaxed text-text-dim">
                        {backend.description}
                      </span>
                      <span className="mt-1 block font-mono text-[10px] text-text-mute">
                        {backend.artifact}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </Panel>

        <div className="flex flex-col gap-5">
          <Panel title="Resolution" index={1}>
            <div className="p-4">
              <Field
                label="Inference size"
                hint="Lower resolution trades recall on small objects for throughput and VRAM headroom."
              >
                <div className="flex gap-2">
                  {config.supported_imgsz.map((size) => (
                    <Button
                      key={size}
                      variant={size === config.imgsz ? "primary" : "secondary"}
                      disabled={busy}
                      onClick={() => void apply({ imgsz: size })}
                      className="flex-1"
                    >
                      {size}px
                    </Button>
                  ))}
                </div>
              </Field>
            </div>
          </Panel>

          <Panel title="Reliability drill" index={2}>
            <div className="flex flex-col gap-3 p-4">
              <p className="text-[12px] leading-relaxed text-text-dim">
                Forces one step down the degradation ladder, exactly as a CUDA out-of-memory
                event would. Use it to prove the fallback path works instead of trusting that
                it does.
              </p>
              <Button variant="danger" onClick={() => void drill()} disabled={busy}>
                <Warning size={15} />
                Trigger degradation
              </Button>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

