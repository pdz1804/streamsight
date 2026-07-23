export function Button({
  variant = "secondary",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger";
}) {
  const base =
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius)] px-3.5 py-2 text-[13px] font-medium transition-[background-color,color,border-color,box-shadow,transform] duration-200 active:translate-y-px disabled:pointer-events-none disabled:opacity-40";
  const variants = {
    primary:
      "bg-accent text-accent-fg shadow-[var(--shadow-sm)] hover:bg-accent-hover hover:shadow-[var(--shadow)]",
    secondary:
      "border border-line bg-surface text-text shadow-[var(--shadow-sm)] hover:border-line-strong hover:bg-surface-2",
    danger: "border border-danger/40 bg-danger/8 text-danger hover:bg-danger/14",
  } as const;
  return <button className={`${base} ${variants[variant]} ${className}`} {...props} />;
}
