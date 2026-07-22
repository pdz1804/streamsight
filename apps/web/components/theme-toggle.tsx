"use client";

/**
 * Three-state theme control: follow the system, force light, force dark.
 *
 * A plain two-way toggle would silently strand users who want to track their OS
 * preference, so "System" stays a first-class option rather than an implicit
 * default you lose the moment you touch the switch.
 */

import { Desktop, Moon, Sun } from "@phosphor-icons/react/dist/ssr";

import { useTheme, type ThemeChoice } from "@/lib/theme";

const OPTIONS: { value: ThemeChoice; label: string; icon: typeof Sun }[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "system", label: "System", icon: Desktop },
  { value: "dark", label: "Dark", icon: Moon },
];

export function ThemeToggle() {
  const { choice, setChoice } = useTheme();

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="flex items-center gap-0.5 rounded-[var(--radius)] border border-line bg-surface-2 p-0.5"
    >
      {OPTIONS.map(({ value, label, icon: Icon }) => {
        const active = choice === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={label}
            title={label}
            onClick={() => setChoice(value)}
            className={`inline-flex size-7 items-center justify-center rounded-[6px] transition-colors ${
              active
                ? "bg-surface text-text shadow-[var(--shadow)]"
                : "text-text-mute hover:text-text"
            }`}
          >
            <Icon size={15} weight={active ? "fill" : "regular"} />
          </button>
        );
      })}
    </div>
  );
}
