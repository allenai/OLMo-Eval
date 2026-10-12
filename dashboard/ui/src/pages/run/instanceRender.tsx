/**
 * Adaptive renderer for one prediction record: prompt (with few-shot collapsed or chat turns),
 * gold label, multiple-choice table, generation output (raw / markdown / code), thinking traces,
 * code execution results, judge verdicts and agent trajectories.
 */
import type { JsonObject } from "@contract/api-types";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import { Check, ChevronRight, CircleCheck, CircleX } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge, Segmented } from "@/components/primitives";
import { wordDiff } from "@/lib/diff";
import { formatCount, sig3 } from "@/lib/format";
import s from "./run.module.css";

hljs.registerLanguage("python", python);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("json", json);

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown): string | null => (typeof v === "string" ? v : v == null ? null : JSON.stringify(v));

export interface ParsedRecord {
  requestType: string | null;
  promptText: string | null;
  fewshot: string | null;
  messages: { role: string; content: string }[] | null;
  gold: string | null;
  choices: { text: string; logprob: number | null; perChar: number | null; bpb: number | null; chosen: boolean; gold: boolean }[] | null;
  output: string | null;
  thinking: string | null;
  extracted: string | null;
  finish: string | null;
  tokens: number | null;
  execution: Obj | null;
  judge: Obj | string | null;
  trajectory: Obj | null;
  scoringErrors: string[];
}

export function parseRecord(prediction: JsonObject | null, request: JsonObject | null): ParsedRecord {
  const req = isObj(request) ? request : {};
  const inner = isObj(req.request) ? req.request : {};
  const doc = isObj(req.doc) ? req.doc : {};
  const pred = isObj(prediction) ? prediction : {};
  const outputs = Array.isArray(pred.model_output) ? (pred.model_output as Obj[]) : [];
  const context = inner.context;
  let promptText: string | null = null;
  let fewshot: string | null = null;
  let messages: ParsedRecord["messages"] = null;
  if (Array.isArray(context)) {
    messages = (context as Obj[]).map((m) => ({ role: String(m.role ?? "user"), content: str(m.content) ?? "" }));
  } else if (typeof context === "string") {
    const blocks = context.split(/\n\n(?=Question:)/);
    if (blocks.length > 1) {
      fewshot = blocks.slice(0, -1).join("\n\n");
      promptText = blocks[blocks.length - 1];
    } else promptText = context;
  } else if (typeof doc.query === "string") promptText = doc.query;

  const choicesRaw = Array.isArray(doc.choices) ? (doc.choices as unknown[]).map((c) => String(c)) : null;
  const goldIdx = typeof pred.label === "number" ? pred.label : typeof doc.gold === "number" ? doc.gold : null;
  let choices: ParsedRecord["choices"] = null;
  if (choicesRaw && outputs.length === choicesRaw.length) {
    const greedy = outputs.findIndex((o) => o.is_greedy === true);
    const predicted = typeof pred.predicted_index_per_char === "number" ? pred.predicted_index_per_char : greedy;
    choices = choicesRaw.map((text, i) => ({
      text,
      logprob: typeof outputs[i].sum_logits === "number" ? (outputs[i].sum_logits as number) : null,
      perChar: typeof outputs[i].logits_per_char === "number" ? (outputs[i].logits_per_char as number) : null,
      bpb: typeof outputs[i].bits_per_byte === "number" ? (outputs[i].bits_per_byte as number) : null,
      chosen: i === predicted,
      gold: i === goldIdx,
    }));
  }
  const first = outputs[0] ?? {};
  const original = str(first.original_text);
  const thinkMatch = original ? /<think>([\s\S]*?)<\/think>/.exec(original) : null;
  const gold = choices ? (goldIdx != null ? choicesRaw![goldIdx] : null) : str(pred.label ?? req.label ?? doc.answer);
  const judge = (isObj(first.judge_result) ? first.judge_result : null) ?? (pred.judge_result != null ? (isObj(pred.judge_result) ? pred.judge_result : String(pred.judge_result)) : null);
  return {
    requestType: str(req.request_type),
    promptText,
    fewshot,
    messages,
    gold,
    choices,
    output: choices ? null : str(first.text),
    thinking: thinkMatch ? thinkMatch[1].trim() : null,
    extracted: choices ? null : str(first.extracted_answer),
    finish: str(first.finish_reason),
    tokens: typeof first.completion_tokens === "number" ? first.completion_tokens : null,
    execution: isObj(first.execution_result) ? first.execution_result : null,
    judge,
    trajectory: isObj(pred.trajectory) ? pred.trajectory : null,
    scoringErrors: Array.isArray(first.scoring_errors) ? (first.scoring_errors as unknown[]).map(String) : [],
  };
}

export function Section({ label, extra, children }: { label: ReactNode; extra?: ReactNode; children: ReactNode }) {
  return (
    <div className={s.section}>
      <div className={s.sectionLabel}>
        {label}
        {extra && <span style={{ marginLeft: "auto", textTransform: "none", letterSpacing: 0, fontWeight: 400 }}>{extra}</span>}
      </div>
      {children}
    </div>
  );
}

function Collapsible({ summary, children, defaultOpen = false }: { summary: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="row"
        style={{ border: 0, background: "transparent", padding: "2px 0", cursor: "pointer", color: "var(--muted)", fontSize: 12, gap: 4 }}
      >
        <ChevronRight size={13} style={{ transform: open ? "rotate(90deg)" : undefined, transition: "transform 0.1s" }} />
        {summary}
      </button>
      {open && <div style={{ marginTop: 6 }}>{children}</div>}
    </div>
  );
}

function approxTokens(text: string): number {
  return Math.round(text.length / 4);
}

export function PromptView({ rec }: { rec: ParsedRecord }) {
  if (rec.messages) {
    const system = rec.messages.filter((m) => m.role === "system");
    const rest = rec.messages.filter((m) => m.role !== "system");
    return (
      <div className="col" style={{ gap: 0 }}>
        {system.length > 0 && (
          <Collapsible summary={`system prompt (${formatCount(approxTokens(system.map((m) => m.content).join(" ")))} tokens)`}>
            <div className={s.text}>{system.map((m) => m.content).join("\n\n")}</div>
          </Collapsible>
        )}
        {rest.map((m, i) => (
          <div key={i} className={s.turn}>
            <span className={s.role}>{m.role}</span>
            <div style={{ whiteSpace: "pre-wrap", fontSize: 13, lineHeight: "20px", overflowWrap: "anywhere" }}>{m.content}</div>
          </div>
        ))}
      </div>
    );
  }
  return (
    <div className="col" style={{ gap: 6 }}>
      {rec.fewshot && (
        <Collapsible summary={`show few-shot (${rec.fewshot.split(/\n\n(?=Question:)/).length} examples, ~${formatCount(approxTokens(rec.fewshot))} tokens)`}>
          <div className={s.text} style={{ color: "var(--muted)" }}>
            {rec.fewshot}
          </div>
        </Collapsible>
      )}
      <div className={s.text}>{rec.promptText ?? "—"}</div>
    </div>
  );
}

export function ChoicesTable({ choices }: { choices: NonNullable<ParsedRecord["choices"]> }) {
  const probs = useMemo(() => {
    const vals = choices.map((c) => c.perChar ?? c.logprob ?? -Infinity);
    const max = Math.max(...vals);
    const exps = vals.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((e) => e / sum);
  }, [choices]);
  return (
    <div className={s.choices}>
      <div className={`${s.choice} ${s.choiceHead}`}>
        <span />
        <span>choice</span>
        <span className="num">logprob</span>
        <span className="num">per char</span>
        <span className="num">bpb</span>
        <span>prob</span>
      </div>
      {choices.map((c, i) => (
        <div key={i} className={`${s.choice} ${c.chosen ? s.choiceChosen : ""}`}>
          <span title={c.gold ? "Gold" : undefined}>{c.gold && <Check size={15} color="var(--status-complete)" />}</span>
          <span style={{ overflowWrap: "anywhere" }}>
            {c.text}
            {c.chosen && (
              <span style={{ marginLeft: 6 }}>
                <Badge tone="teal">chosen</Badge>
              </span>
            )}
          </span>
          <span className="num">{c.logprob != null ? c.logprob.toFixed(2) : "—"}</span>
          <span className="num">{c.perChar != null ? c.perChar.toFixed(3) : "—"}</span>
          <span className="num">{c.bpb != null ? sig3(c.bpb) : "—"}</span>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ flex: 1, height: 6, borderRadius: 3, background: "var(--row)", overflow: "hidden" }}>
              <span style={{ display: "block", width: `${probs[i] * 100}%`, height: "100%", background: c.chosen ? "var(--teal-line)" : "var(--seq-2)" }} />
            </span>
            <span className="mono muted" style={{ fontSize: 10.5, width: 30, textAlign: "right" }}>
              {Math.round(probs[i] * 100)}%
            </span>
          </span>
        </div>
      ))}
    </div>
  );
}

function looksLikeCode(text: string): boolean {
  return /```|^\s*(def |class |import |from |#include|function )/m.test(text);
}

export function OutputText({ text, mode }: { text: string; mode: "raw" | "md" | "code" }) {
  if (mode === "md") {
    return (
      <div className={s.prose}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>
    );
  }
  if (mode === "code") {
    const code = text.replace(/^```\w*\n?/, "").replace(/\n?```\s*$/, "");
    const html = hljs.highlightAuto(code, ["python", "bash", "json"]).value;
    return <pre className={s.text} dangerouslySetInnerHTML={{ __html: html }} />;
  }
  return <div className={s.text}>{text}</div>;
}

export function DiffText({ a, b }: { a: string; b: string }) {
  const segments = useMemo(() => wordDiff(a, b), [a, b]);
  return (
    <div className={s.text}>
      {segments.map((seg, i) =>
        seg.kind === "same" ? (
          <span key={i}>{seg.text}</span>
        ) : (
          <span key={i} className={seg.kind === "ins" ? s.ins : s.del}>
            {seg.text}
          </span>
        ),
      )}
    </div>
  );
}

export function OutputView({ rec, diffAgainst, compact }: { rec: ParsedRecord; diffAgainst?: string | null; compact?: boolean }) {
  const [mode, setMode] = useState<"raw" | "md" | "code">(() => (rec.output && looksLikeCode(rec.output) ? "code" : "raw"));
  if (rec.output == null) return <span className="t-caption">No generated text for this request type.</span>;
  return (
    <div className="col" style={{ gap: 6 }}>
      {rec.thinking && (
        <Collapsible summary={`thinking trace (${formatCount(approxTokens(rec.thinking))} tokens)`}>
          <div className={s.text} style={{ color: "var(--muted)" }}>
            {rec.thinking}
          </div>
        </Collapsible>
      )}
      {!compact && diffAgainst == null && (
        <div className="row">
          <span className="spacer" />
          <Segmented
            value={mode}
            onChange={setMode}
            label="Render mode"
            options={[
              { value: "raw", label: "raw" },
              { value: "md", label: "markdown" },
              { value: "code", label: "code" },
            ]}
          />
        </div>
      )}
      {diffAgainst != null ? <DiffText a={diffAgainst} b={rec.output} /> : <OutputText text={rec.output} mode={mode} />}
      <div className="row-wrap t-caption" style={{ gap: 12 }}>
        {rec.extracted != null && (
          <span>
            extracted <span className="mono" style={{ color: "var(--fg)" }}>{rec.extracted.length > 80 ? `${rec.extracted.slice(0, 80)}…` : rec.extracted}</span>
          </span>
        )}
        {rec.finish && (
          <span>
            finish <span className="mono" style={{ color: rec.finish === "length" ? "var(--warn-fg)" : "var(--fg)" }}>{rec.finish}</span>
          </span>
        )}
        {rec.tokens != null && (
          <span>
            tokens <span className="mono" style={{ color: "var(--fg)" }}>{formatCount(rec.tokens)}</span>
          </span>
        )}
      </div>
    </div>
  );
}

export function ExecutionView({ exec }: { exec: Obj }) {
  const ok = exec.success === true;
  const output = str(exec.output) ?? "";
  const [full, setFull] = useState(false);
  const shown = full ? output : output.slice(0, 20000);
  return (
    <div className="col" style={{ gap: 6 }}>
      <div className="row" style={{ gap: 8, fontSize: 13 }}>
        {ok ? <CircleCheck size={16} color="var(--status-complete)" /> : <CircleX size={16} color="var(--status-failed)" />}
        <strong>{ok ? "Tests passed" : "Tests failed"}</strong>
        {exec.num_tests != null && <span className="t-caption">{String(exec.num_tests)} tests</span>}
        {exec.exit_code != null && <span className="t-caption mono">exit {String(exec.exit_code)}</span>}
      </div>
      {str(exec.error) && <div className={s.text} style={{ color: "var(--danger-fg)", maxHeight: 120 }}>{str(exec.error)}</div>}
      {output && (
        <>
          <div className={s.text} style={{ maxHeight: 220 }}>
            {shown}
          </div>
          {output.length > 20000 && !full && (
            <button type="button" className="link t-caption" onClick={() => setFull(true)} style={{ border: 0, background: "transparent" }}>
              load full output
            </button>
          )}
        </>
      )}
    </div>
  );
}

export function JudgeView({ judge }: { judge: Obj | string }) {
  if (typeof judge === "string") {
    return (
      <div className="row" style={{ gap: 8 }}>
        <Badge tone={/correct|safe|win/i.test(judge) && !/incorrect|unsafe/i.test(judge) ? "teal" : "warn"}>{judge}</Badge>
      </div>
    );
  }
  return (
    <div className="col" style={{ gap: 6, fontSize: 13 }}>
      <div className="row" style={{ gap: 8 }}>
        {judge.verdict != null && <Badge tone="teal">{String(judge.verdict)}</Badge>}
        {judge.score != null && <span className="mono">score {String(judge.score)}</span>}
        {judge.judge_model != null && <span className="t-caption">by {String(judge.judge_model)}</span>}
      </div>
      {judge.rationale != null && <div className={s.prose} style={{ fontSize: 13 }}>{String(judge.rationale)}</div>}
    </div>
  );
}

export function TrajectoryView({ trajectory }: { trajectory: Obj }) {
  const turns = Array.isArray(trajectory.turns) ? (trajectory.turns as Obj[]) : [];
  const meta = isObj(trajectory.metadata) ? trajectory.metadata : {};
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="t-caption">
        {turns.length} turns{meta.max_turns != null ? ` of ${String(meta.max_turns)} max` : ""}
        {meta.hit_max_turns === true && <Badge tone="warn">hit max turns</Badge>}
      </div>
      <div className={s.timeline}>
        {turns.map((t, i) => (
          <div key={i} className="col" style={{ gap: 4 }}>
            <span className={s.role}>
              {String(t.role)}
              {t.token_count != null && <span style={{ fontWeight: 400, marginLeft: 6 }}>{String(t.token_count)} tok</span>}
            </span>
            {str(t.content) && <div style={{ fontSize: 13, whiteSpace: "pre-wrap" }}>{str(t.content)}</div>}
            {Array.isArray(t.tool_calls) &&
              (t.tool_calls as Obj[]).map((call, j) => (
                <details key={j} className={s.toolCard}>
                  <summary>
                    <span style={{ color: "var(--cat-5)" }}>call</span> {String(call.name)}
                  </summary>
                  <pre>{JSON.stringify(call.arguments, null, 2)}</pre>
                </details>
              ))}
            {Array.isArray(t.tool_results) &&
              (t.tool_results as Obj[]).map((res, j) => (
                <details key={j} className={s.toolCard}>
                  <summary>
                    <span style={{ color: "var(--better)" }}>result</span> {String(res.name)}
                  </summary>
                  <pre>{JSON.stringify(res.result, null, 2)}</pre>
                </details>
              ))}
          </div>
        ))}
      </div>
      {str(trajectory.final_answer) && (
        <Section label="Final answer">
          <div className={s.prose}>{str(trajectory.final_answer)}</div>
        </Section>
      )}
    </div>
  );
}
