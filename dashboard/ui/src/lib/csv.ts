export type Cell = string | number | boolean | null | undefined;

function csvEscape(value: Cell): string {
  if (value == null) return "";
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function tsvEscape(value: Cell): string {
  if (value == null) return "";
  return String(value).replace(/[\t\n\r]+/g, " ");
}

export function toCSV(header: string[], rows: Cell[][]): string {
  return [header, ...rows].map((r) => r.map(csvEscape).join(",")).join("\n") + "\n";
}

export function toTSV(header: string[], rows: Cell[][]): string {
  return [header, ...rows].map((r) => r.map(tsvEscape).join("\t")).join("\n") + "\n";
}

export function downloadText(filename: string, text: string, mime = "text/csv"): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
}
