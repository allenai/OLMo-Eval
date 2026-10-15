import type { ReactNode } from "react";
import { AppLink } from "./AppLink";
import p from "./page.module.css";

export function PageHeader({
  title,
  crumbs,
  meta,
  actions,
  children,
  titleExtra,
}: {
  title: ReactNode;
  crumbs?: { label: ReactNode; href?: string }[];
  meta?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  titleExtra?: ReactNode;
}) {
  return (
    <header className={p.header}>
      {crumbs && crumbs.length > 0 && (
        <nav className={p.crumbs} aria-label="Breadcrumb">
          {crumbs.map((c, i) => (
            <span key={i} className={p.crumb}>
              {c.href ? <AppLink href={c.href}>{c.label}</AppLink> : c.label}
              <span className={p.crumbSep}>/</span>
            </span>
          ))}
        </nav>
      )}
      <div className={p.titleRow}>
        <h1 className={p.title}>{title}</h1>
        {titleExtra}
        <span className="spacer" />
        {actions && <div className={p.actions}>{actions}</div>}
      </div>
      {meta && <div className={p.meta}>{meta}</div>}
      {children}
    </header>
  );
}

export function MetaItem({ label, children }: { label?: ReactNode; children: ReactNode }) {
  return (
    <span className={p.metaItem}>
      {label && <span className={p.metaLabel}>{label}</span>}
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
  extra,
  onClick,
  title,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  extra?: ReactNode;
  onClick?: () => void;
  title?: string;
}) {
  return (
    <div className={p.stat} onClick={onClick} role={onClick ? "button" : undefined} tabIndex={onClick ? 0 : undefined} title={title}>
      <div className={p.statLabel}>{label}</div>
      <div className={p.statValue}>{value}</div>
      {sub && <div className={p.statSub}>{sub}</div>}
      {extra}
    </div>
  );
}
