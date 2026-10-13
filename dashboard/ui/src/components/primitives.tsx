import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import * as RadixTooltip from "@radix-ui/react-tooltip";
import {
  AlertTriangle,
  ArrowUpFromLine,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Copy,
  ExternalLink,
  Info,
  Loader2,
  Search,
  X,
  XCircle,
} from "lucide-react";
import {
  type ButtonHTMLAttributes,
  type CSSProperties,
  forwardRef,
  type InputHTMLAttributes,
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import type { RunStatus, RunSummary } from "@contract/api-types";
import { ApiError } from "@/api/client";
import { copyText } from "@/lib/csv";
import { absoluteLocal, relativeTime, utcString } from "@/lib/time";
import { catColor } from "@/lib/scales";
import s from "./ui.module.css";
import { toast } from "./toast";
import { pressable } from "@/lib/pressable";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

// ---------------------------------------------------------------- buttons

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "ghost" | "danger";
  size?: "md" | "sm";
  icon?: ReactNode;
  pressed?: boolean;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "default", size = "md", icon, pressed, className, children, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx(
        s.button,
        variant === "primary" && s.primary,
        variant === "ghost" && s.ghost,
        variant === "danger" && s.danger,
        size === "sm" && s.sm,
        pressed && s.pressed,
        className,
      )}
      aria-pressed={pressed}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
});

type IconButtonProps = Omit<ButtonProps, "children" | "icon"> & { label: string; icon: ReactNode };

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, variant = "ghost", size = "md", className, ...rest },
  ref,
) {
  return (
    <Button
      ref={ref}
      variant={variant}
      size={size}
      className={cx(s.iconButton, className)}
      aria-label={label}
      title={label}
      {...rest}
    >
      {icon}
    </Button>
  );
});

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <div className={s.segmented} role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={s.segment}
          aria-pressed={o.value === value}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- inputs

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...rest },
  ref,
) {
  return <input ref={ref} className={cx(s.input, className)} {...rest} />;
});

export const SearchInput = forwardRef<
  HTMLInputElement,
  InputHTMLAttributes<HTMLInputElement> & { wrapStyle?: CSSProperties }
>(function SearchInput({ className, wrapStyle, ...rest }, ref) {
  return (
    <div className={s.inputWrap} style={wrapStyle}>
      <Search />
      <input ref={ref} type="search" className={cx(s.input, className)} {...rest} />
    </div>
  );
});

/**
 * A search box that keeps its text locally and commits it after the user stops typing (or on
 * Enter), so a server-side search runs once per pause instead of once per keystroke.
 */
export function DebouncedSearchInput({
  value,
  onCommit,
  delay = 250,
  ...rest
}: Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange"> & {
  value: string;
  onCommit: (value: string) => void;
  delay?: number;
  wrapStyle?: CSSProperties;
}) {
  const [text, setText] = useState(value);
  const [synced, setSynced] = useState(value);
  if (value !== synced) {
    // The committed value changed elsewhere (navigation, a cleared filter): show it.
    setSynced(value);
    setText(value);
  }
  const commit = useRef(onCommit);
  useEffect(() => {
    commit.current = onCommit;
  });
  useEffect(() => {
    if (text === value) return;
    const t = setTimeout(() => commit.current(text), delay);
    return () => clearTimeout(t);
  }, [text, value, delay]);
  return (
    <SearchInput
      {...rest}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter" && text !== value) commit.current(text);
        rest.onKeyDown?.(e);
      }}
    />
  );
}

export function Checkbox({
  checked,
  indeterminate,
  onChange,
  label,
  ariaLabel,
  disabled,
}: {
  checked: boolean;
  indeterminate?: boolean;
  onChange: (checked: boolean, shiftKey: boolean) => void;
  label?: ReactNode;
  ariaLabel?: string;
  disabled?: boolean;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const shift = useRef(false);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = !!indeterminate;
  }, [indeterminate]);
  return (
    <label className={s.check} onClick={(e) => e.stopPropagation()}>
      <input
        ref={ref}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        aria-label={ariaLabel}
        onMouseDown={(e) => (shift.current = e.shiftKey)}
        onKeyDown={(e) => (shift.current = e.shiftKey)}
        onChange={(e) => onChange(e.target.checked, shift.current)}
      />
      {label}
    </label>
  );
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  ariaLabel,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  ariaLabel: string;
}) {
  return (
    <input
      className={s.slider}
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      aria-label={ariaLabel}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

// ---------------------------------------------------------------- select (dropdown)

export interface SelectOption<T extends string> {
  value: T;
  label: ReactNode;
  hint?: ReactNode;
  group?: string;
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  label,
  width,
  size,
  renderValue,
}: {
  value: T;
  options: SelectOption<T>[];
  onChange: (v: T) => void;
  label?: string;
  width?: number | string;
  size?: "sm" | "md";
  renderValue?: (opt: SelectOption<T> | undefined) => ReactNode;
}) {
  const current = options.find((o) => o.value === value);
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <button type="button" className={cx(s.select, size === "sm" && s.sm)} style={{ width }} aria-label={label}>
          {label && <span className={s.selectLabel}>{label}</span>}
          <span className={s.selectValue}>{renderValue ? renderValue(current) : (current?.label ?? value)}</span>
          <ChevronDown />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={s.menu} align="start" sideOffset={4} collisionPadding={8}>
          {options.map((o, i) => {
            const header = o.group && (i === 0 || options[i - 1].group !== o.group) ? o.group : null;
            return (
              <div key={o.value}>
                {header && <DropdownMenu.Label className={s.menuLabel}>{header}</DropdownMenu.Label>}
                <DropdownMenu.Item className={s.menuItem} onSelect={() => onChange(o.value)}>
                  <span style={{ width: 15, display: "inline-flex" }}>{o.value === value && <Check />}</span>
                  <span className="truncate">{o.label}</span>
                  {o.hint != null && <span className={s.menuItemHint}>{o.hint}</span>}
                </DropdownMenu.Item>
              </div>
            );
          })}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

// ---------------------------------------------------------------- menu

export interface MenuItemSpec {
  label?: ReactNode;
  icon?: ReactNode;
  hint?: ReactNode;
  onSelect?: () => void;
  href?: string;
  disabled?: boolean;
  separator?: boolean;
  heading?: string;
}

export function Menu({
  trigger,
  items,
  align = "end",
}: {
  trigger: ReactNode;
  items: MenuItemSpec[];
  align?: "start" | "end" | "center";
}) {
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={s.menu} align={align} sideOffset={4} collisionPadding={8}>
          {items.map((item, i) => {
            if (item.separator) return <DropdownMenu.Separator key={i} className={s.menuSep} />;
            if (item.heading) return <DropdownMenu.Label key={i} className={s.menuLabel}>{item.heading}</DropdownMenu.Label>;
            return (
              <DropdownMenu.Item
                key={i}
                className={s.menuItem}
                disabled={item.disabled}
                onSelect={() => {
                  if (item.href) window.open(item.href, "_blank", "noopener");
                  item.onSelect?.();
                }}
              >
                {item.icon}
                <span className="truncate">{item.label}</span>
                {item.hint != null && <span className={s.menuItemHint}>{item.hint}</span>}
              </DropdownMenu.Item>
            );
          })}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

// ---------------------------------------------------------------- tooltip

export function Tip({
  content,
  children,
  side = "top",
  delay = 250,
}: {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  delay?: number;
}) {
  if (content == null || content === "") return <>{children}</>;
  return (
    <RadixTooltip.Root delayDuration={delay}>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content className={s.tooltip} side={side} sideOffset={6} collisionPadding={8}>
          {content}
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}

// ---------------------------------------------------------------- chips and badges

export function Chip({
  label,
  value,
  onRemove,
  active,
  onClick,
  title,
}: {
  label?: ReactNode;
  value: ReactNode;
  onRemove?: () => void;
  active?: boolean;
  onClick?: () => void;
  title?: string;
}) {
  return (
    <span
      className={cx(s.chip, active && s.chipActive)}
      title={title}
      {...(onClick ? pressable(onClick) : {})}
      aria-pressed={onClick && active !== undefined ? active : undefined}
      style={onClick ? { cursor: "pointer" } : undefined}
    >
      <span>
        {label && <span className={s.chipKey}>{label}: </span>}
        {value}
      </span>
      {onRemove && (
        <button
          type="button"
          className={s.chipRemove}
          aria-label="Remove"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
        >
          <X />
        </button>
      )}
      {!onRemove && <span style={{ width: 5 }} />}
    </span>
  );
}

type BadgeTone = "teal" | "muted" | "warn" | "danger" | "outline" | "base";

export function Badge({ children, tone = "teal", title }: { children: ReactNode; tone?: BadgeTone; title?: string }) {
  return (
    <span
      title={title}
      className={cx(
        s.badge,
        tone === "muted" && s.badgeMuted,
        tone === "warn" && s.badgeWarn,
        tone === "danger" && s.badgeDanger,
        tone === "outline" && s.badgeOutline,
        tone === "base" && s.badgeBase,
      )}
    >
      {children}
    </span>
  );
}

export function BaseTag() {
  return (
    <Badge tone="base" title="Baseline">
      BASE
    </Badge>
  );
}

export type DisplayStatus = RunStatus | "uploading" | "stale";

export function displayStatus(run: Pick<RunSummary, "status" | "stale" | "upload_state">): DisplayStatus {
  if (run.status === "running") return run.stale ? "stale" : "running";
  if (run.upload_state === "uploading") return "uploading";
  return run.status;
}

const STATUS_META: Record<DisplayStatus, { label: string; color: string; icon: ReactNode }> = {
  complete: { label: "Complete", color: "var(--status-complete)", icon: <CheckCircle2 /> },
  partial: { label: "Partial", color: "var(--status-partial)", icon: <HalfCircle /> },
  failed: { label: "Failed", color: "var(--status-failed)", icon: <XCircle /> },
  running: { label: "Running", color: "var(--status-running)", icon: <Loader2 className={s.spin} /> },
  stale: { label: "Stale", color: "var(--muted)", icon: <CircleDashed /> },
  uploading: { label: "Uploading", color: "var(--status-uploading)", icon: <ArrowUpFromLine /> },
};

function HalfCircle() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" />
    </svg>
  );
}

export function StatusBadge({ status, iconOnly }: { status: DisplayStatus; iconOnly?: boolean }) {
  const meta = STATUS_META[status];
  const title =
    status === "stale" ? "Running, but no update for over 24 hours. The job may have crashed." : meta.label;
  return (
    <span className={cx(s.status, iconOnly && s.statusIconOnly)} title={title} aria-label={meta.label}>
      <span style={{ color: meta.color, display: "inline-flex" }}>{meta.icon}</span>
      {!iconOnly && meta.label}
    </span>
  );
}

// ---------------------------------------------------------------- panels

export function Panel({
  title,
  caption,
  actions,
  children,
  pad = "default",
  refetching,
  className,
  style,
  id,
}: {
  title?: ReactNode;
  caption?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  pad?: "default" | "chart" | "none";
  refetching?: boolean;
  className?: string;
  style?: CSSProperties;
  id?: string;
}) {
  return (
    <section
      id={id}
      className={cx(s.panel, pad === "default" && s.panelPad, pad === "chart" && s.panelChart, className)}
      style={style}
    >
      {refetching && <div className={s.refetchBar} aria-hidden />}
      {(title || actions) && (
        <header className={s.panelHeader} style={pad === "none" ? { padding: "12px 12px 0" } : undefined}>
          <div className={s.panelTitles}>
            {title && <h2 className={s.panelTitle}>{title}</h2>}
            {caption && <p className={s.panelCaption}>{caption}</p>}
          </div>
          {actions && <div className={s.panelActions}>{actions}</div>}
        </header>
      )}
      <div className={refetching ? s.dimmed : undefined}>{children}</div>
    </section>
  );
}

// ---------------------------------------------------------------- tabs

export interface TabSpec<T extends string> {
  value: T;
  label: ReactNode;
  count?: number | null;
  hidden?: boolean;
}

export function Tabs<T extends string>({
  value,
  tabs,
  onChange,
  label,
  showKeys,
}: {
  value: T;
  tabs: TabSpec<T>[];
  onChange: (v: T) => void;
  label: string;
  showKeys?: boolean;
}) {
  const visibleTabs = tabs.filter((t) => !t.hidden);
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const idx = visibleTabs.findIndex((t) => t.value === value);
    const next = visibleTabs[(idx + (e.key === "ArrowRight" ? 1 : -1) + visibleTabs.length) % visibleTabs.length];
    onChange(next.value);
    e.preventDefault();
  };
  return (
    <div className={s.tabs} role="tablist" aria-label={label} onKeyDown={onKeyDown}>
      {visibleTabs.map((t, i) => (
        <button
          key={t.value}
          type="button"
          role="tab"
          className={s.tab}
          aria-selected={t.value === value}
          tabIndex={t.value === value ? 0 : -1}
          onClick={() => onChange(t.value)}
        >
          {t.label}
          {t.count != null && <span className={s.tabCount}>{t.count}</span>}
          {showKeys && i < 9 && <span className={cx(s.tabKey, "hide-md")}>{i + 1}</span>}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- states

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className={s.kbd}>{children}</kbd>;
}

export function Skeleton({ width = "100%", height = 12, style }: { width?: number | string; height?: number; style?: CSSProperties }) {
  return <span className={s.skeleton} style={{ width, height, ...style }} aria-hidden />;
}

export function SkeletonRows({ rows = 8, columns = 6 }: { rows?: number; columns?: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: "10px 12px" }} aria-busy aria-label="Loading">
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} style={{ display: "grid", gridTemplateColumns: `2fr repeat(${columns - 1}, 1fr)`, gap: 16 }}>
          {Array.from({ length: columns }, (_, c) => (
            <Skeleton key={c} width={c === 0 ? "80%" : `${50 + ((r * 7 + c * 13) % 40)}%`} />
          ))}
        </div>
      ))}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  actions,
  compact,
}: {
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
  compact?: boolean;
}) {
  return (
    <div className={s.empty} style={compact ? { padding: "18px 12px" } : undefined}>
      {icon}
      <div className={s.emptyTitle}>{title}</div>
      {children && <div className={s.emptyBody}>{children}</div>}
      {actions && <div className="row" style={{ marginTop: 8 }}>{actions}</div>}
    </div>
  );
}

export function ErrorPanel({ error, onRetry, title }: { error: unknown; onRetry?: () => void; title?: string }) {
  const api = error instanceof ApiError ? error : null;
  const auth = api && (api.status === 401 || api.status === 403) && api.code !== "forbidden";
  const notFound = api?.status === 404;
  return (
    <div className={s.error} role="alert">
      <AlertTriangle />
      <div className={s.errorBody}>
        <strong>
          {title ?? (auth ? "Your session expired" : notFound ? "Not found" : "Something went wrong loading this panel")}
        </strong>
        <span>{auth ? "Reload the page to sign in again." : api?.message ?? (error instanceof Error ? error.message : String(error))}</span>
        <div className={s.errorMeta}>
          {api && api.status > 0 && <span>HTTP {api.status}</span>}
          {api?.code && <span>{api.code}</span>}
          {api?.requestId && (
            <span>
              request <CopyText text={api.requestId} />
            </span>
          )}
        </div>
      </div>
      {auth ? (
        <Button size="sm" onClick={() => window.location.reload()}>
          Reload
        </Button>
      ) : (
        onRetry && (
          <Button size="sm" onClick={onRetry}>
            Retry
          </Button>
        )
      )}
    </div>
  );
}

export function Banner({ children, tone = "warn", icon }: { children: ReactNode; tone?: "warn" | "info"; icon?: ReactNode }) {
  return (
    <div className={cx(s.banner, tone === "info" && s.bannerInfo)}>
      {icon ?? (tone === "warn" ? <AlertTriangle /> : <Info />)}
      <div style={{ minWidth: 0 }}>{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------- small pieces

export function CopyText({
  text,
  display,
  className,
  title,
}: {
  text: string;
  display?: ReactNode;
  className?: string;
  title?: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className={cx(s.copy, copied && s.copied, className)}
      title={title ?? `Copy ${text}`}
      onClick={async (e) => {
        e.stopPropagation();
        if (await copyText(text)) {
          setCopied(true);
          toast("Copied to clipboard");
          setTimeout(() => setCopied(false), 1200);
        }
      }}
    >
      <span className="truncate">{display ?? text}</span>
      {copied ? <Check /> : <Copy />}
    </button>
  );
}

export function RelativeTime({ value, className }: { value: string | null | undefined; className?: string }) {
  if (!value) return <span className={className}>—</span>;
  return (
    <Tip
      content={
        <span>
          {absoluteLocal(value)}
          <br />
          <span className="muted">{utcString(value)}</span>
        </span>
      }
    >
      <time dateTime={value} className={cx("nowrap", className)}>
        {relativeTime(value)}
      </time>
    </Tip>
  );
}

export function ModelDot({ slot, baseline, title }: { slot?: number | null; baseline?: boolean; title?: string }) {
  if (baseline) return <span className={cx(s.dot, s.dotBase)} title={title ?? "Baseline"} />;
  return <span className={s.dot} style={{ background: catColor(slot) }} title={title} />;
}

export function LinkOut({ href, children, title }: { href: string | null | undefined; children: ReactNode; title?: string }) {
  if (!href) return <span className="muted">{children}</span>;
  return (
    <a className={s.linkOut} href={href} target="_blank" rel="noopener noreferrer" title={title} onClick={(e) => e.stopPropagation()}>
      {children}
      <ExternalLink />
    </a>
  );
}

export function KV({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className={s.kv}>
      {items.map(([k, v], i) => (
        <div key={i} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v ?? <span className="faint">—</span>}</dd>
        </div>
      ))}
    </dl>
  );
}

export { s as uiStyles };
