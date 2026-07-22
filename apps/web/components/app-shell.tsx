"use client";

/**
 * Application chrome: brand, navigation, live backend status, theme control.
 *
 * The shell owns the viewport height and `main` is the only scroll container, so
 * pages never produce a second scrollbar next to the body's.
 */

import {
  ChartLineUp,
  Broadcast,
  ImageSquare,
  SlidersHorizontal,
} from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

import { ThemeToggle } from "./theme-toggle";
import { Chip } from "./ui";

const NAV = [
  { href: "/", label: "Live", icon: Broadcast },
  { href: "/upload", label: "Single frame", icon: ImageSquare },
  { href: "/metrics", label: "Metrics", icon: ChartLineUp },
  { href: "/settings", label: "Settings", icon: SlidersHorizontal },
] as const;

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="relative z-10 flex h-[100dvh] flex-col overflow-hidden">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 focus:rounded-[var(--radius)] focus:bg-accent focus:px-3 focus:py-2 focus:text-[13px] focus:text-accent-fg"
      >
        Skip to content
      </a>

      <header className="flex h-16 shrink-0 items-center gap-4 border-b border-line bg-surface/85 px-4 backdrop-blur-md md:px-6">
        <Link href="/" className="flex items-center gap-2.5" aria-label="StreamSight home">
          <span
            className="grid size-7 place-items-center rounded-[7px] bg-accent text-accent-fg shadow-[var(--shadow-sm)]"
            aria-hidden="true"
          >
            <Broadcast size={15} weight="fill" />
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-text">StreamSight</span>
        </Link>

        <nav
          aria-label="Primary"
          className="ml-3 flex items-center gap-1 rounded-[var(--radius)] border border-line bg-surface-2 p-1"
        >
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={`inline-flex items-center gap-2 whitespace-nowrap rounded-[6px] px-3 py-1.5 text-[13px] font-medium transition-all duration-200 ${
                  active
                    ? "bg-surface text-text shadow-[var(--shadow-sm)]"
                    : "text-text-dim hover:text-text"
                }`}
              >
                <Icon size={16} weight={active ? "fill" : "regular"} />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <BackendStatus />
          <ThemeToggle />
        </div>
      </header>

      <main id="main" className="min-h-0 flex-1 overflow-y-auto">
        {children}
      </main>
    </div>
  );
}

/** Polls health slowly: enough to notice a dead API, cheap enough to ignore. */
function BackendStatus() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const result = await api.health();
        if (cancelled) return;
        setHealth(result);
        setReachable(true);
      } catch {
        if (cancelled) return;
        setReachable(false);
      }
    };
    void check();
    const timer = window.setInterval(check, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  if (reachable === null) return null;
  if (!reachable) {
    return <Chip tone="danger">API offline</Chip>;
  }

  const gpu = health?.gpu;
  const label = gpu?.available ? gpu.name.replace(/^NVIDIA\s+/, "") : "CPU inference";
  return (
    <span className="hidden items-center gap-2 md:inline-flex">
      <Chip tone="ok" live>
        API up
      </Chip>
      <span className="font-mono text-[11px] text-text-mute">{label}</span>
    </span>
  );
}

