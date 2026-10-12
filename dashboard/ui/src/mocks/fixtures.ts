/**
 * Deterministic synthetic dataset for the mock API.
 *
 * Model quality is a per-category "ability" that grows with checkpoint step. Each task has fixed
 * per-instance difficulties, so models agree on which instances are hard, and checkpoints of the
 * same series share most of their per-instance noise (Gaussian copula), which makes paired
 * comparisons between neighboring checkpoints realistic: small deltas, few flips, narrow CIs.
 */
import type {
  BeakerDetail,
  DisplayFormat,
  EnvironmentDetail,
  Headline,
  MetricKind,
  MetricMeta,
  ModelDetail,
  RunLinks,
  RunStatus,
  RunSummary,
  RuntimeBasis,
  SuiteDef,
  UploadState,
} from "@contract/api-types";
import { beakerId, clamp, hashString, hexId, normal, rngFor, sigmoid } from "./random";

export const HOUR = 3_600_000;
export const DAY = 24 * HOUR;
/** Anchor "now" to the hour so the dataset is stable within a session. */
export const NOW = Math.floor(Date.now() / HOUR) * HOUR;

export const RESULTS_BUCKET = "ai2-skiff2-olmo-eval-results";

// ---------------------------------------------------------------------------------------- tasks

export type TaskType =
  | "mc"
  | "math"
  | "qa"
  | "span"
  | "code"
  | "safety"
  | "ifeval"
  | "judge"
  | "omni"
  | "bpb"
  | "agent"
  | "bbh";

export type Category =
  | "knowledge"
  | "commonsense"
  | "math"
  | "code"
  | "safety"
  | "instruct"
  | "lm"
  | "agent"
  | "reading";

export interface TaskDef {
  name: string;
  base: string;
  type: TaskType;
  category: Category;
  kind: MetricKind;
  primary: string;
  metrics: string[];
  meta: Record<string, MetricMeta>;
  n: number;
  difficulty: number;
  spread: number;
  chance: number;
  fewshot: number | null;
  maxTokens: number | null;
  split: string;
  scale: number;
  meanTokens: number;
  suite: string | null;
}

function meta(kind: MetricKind, hib: boolean | null = true, format?: DisplayFormat): MetricMeta {
  return {
    kind,
    higher_is_better: hib,
    display_format: format ?? (kind === "unbounded" ? "raw" : "percent"),
    unit: null,
  };
}

interface TaskSpec {
  name: string;
  type: TaskType;
  category: Category;
  n: number;
  d: number;
  chance?: number;
  suite?: string;
  spread?: number;
}

const TASK_SPECS: TaskSpec[] = [
  // ARC
  { name: "arc_challenge:rc", type: "mc", category: "knowledge", n: 1000, d: 0.5, chance: 0.25, suite: "arc:rc" },
  { name: "arc_easy:rc", type: "mc", category: "knowledge", n: 1000, d: -0.9, chance: 0.25, suite: "arc:rc" },
  // MMLU subjects
  ...(
    [
      ["abstract_algebra", 100, 1.4],
      ["anatomy", 135, 0.2],
      ["astronomy", 152, 0.1],
      ["college_biology", 144, -0.1],
      ["college_chemistry", 100, 1.0],
      ["computer_security", 100, -0.2],
      ["econometrics", 114, 0.9],
      ["high_school_physics", 151, 0.8],
      ["jurisprudence", 108, 0.2],
      ["moral_scenarios", 400, 1.1],
      ["philosophy", 311, 0.2],
      ["world_religions", 171, -0.5],
    ] as const
  ).map(
    ([subject, n, d]): TaskSpec => ({
      name: `mmlu_${subject}:mc`,
      type: "mc",
      category: "knowledge",
      n,
      d,
      chance: 0.25,
      suite: "mmlu:mc",
    }),
  ),
  // Commonsense
  { name: "hellaswag:rc", type: "mc", category: "commonsense", n: 1000, d: -0.4, chance: 0.25, suite: "commonsense:rc" },
  { name: "piqa:rc", type: "mc", category: "commonsense", n: 1000, d: -1.0, chance: 0.5, suite: "commonsense:rc" },
  { name: "winogrande:rc", type: "mc", category: "commonsense", n: 1000, d: -0.1, chance: 0.5, suite: "commonsense:rc" },
  { name: "csqa:rc", type: "mc", category: "commonsense", n: 1000, d: 0.0, chance: 0.2, suite: "commonsense:rc" },
  { name: "socialiqa:rc", type: "mc", category: "commonsense", n: 1000, d: 0.3, chance: 0.33, suite: "commonsense:rc" },
  { name: "openbookqa:rc", type: "mc", category: "commonsense", n: 500, d: 0.4, chance: 0.25, suite: "commonsense:rc" },
  { name: "boolq:rc", type: "mc", category: "commonsense", n: 1000, d: -0.6, chance: 0.5, suite: "commonsense:rc" },
  // Math
  { name: "gsm8k:cot:olmo3", type: "math", category: "math", n: 1319, d: 0.2, suite: "math:olmo3" },
  { name: "minerva_math_algebra", type: "math", category: "math", n: 500, d: 0.6, suite: "math:olmo3" },
  { name: "minerva_math_geometry", type: "math", category: "math", n: 479, d: 1.2, suite: "math:olmo3" },
  { name: "minerva_math_number_theory", type: "math", category: "math", n: 540, d: 1.0, suite: "math:olmo3" },
  { name: "minerva_math_precalculus", type: "math", category: "math", n: 546, d: 1.5, suite: "math:olmo3" },
  { name: "math500", type: "math", category: "math", n: 500, d: 1.1, suite: "math:olmo3" },
  { name: "aime24", type: "math", category: "math", n: 30, d: 2.8, spread: 0.6, suite: "math:olmo3" },
  // Code
  { name: "humaneval:pass@1", type: "code", category: "code", n: 164, d: 0.4, suite: "code:olmo3" },
  { name: "mbpp:pass@1", type: "code", category: "code", n: 500, d: 0.3, suite: "code:olmo3" },
  { name: "livecodebench:v6", type: "code", category: "code", n: 300, d: 1.9, suite: "code:olmo3" },
  { name: "bigcodebench:hard", type: "code", category: "code", n: 148, d: 2.1, suite: "code:olmo3" },
  // Safety
  { name: "harmbench", type: "safety", category: "safety", n: 400, d: -0.4, suite: "safety:olmo3" },
  { name: "xstest", type: "safety", category: "safety", n: 450, d: -0.8, suite: "safety:olmo3" },
  { name: "wildguardtest", type: "safety", category: "safety", n: 1000, d: -0.2, suite: "safety:olmo3" },
  { name: "do_anything_now", type: "safety", category: "safety", n: 500, d: 0.1, suite: "safety:olmo3" },
  // Not in any suite
  { name: "ifeval", type: "ifeval", category: "instruct", n: 541, d: 0.1 },
  { name: "drop", type: "span", category: "reading", n: 1000, d: 0.3 },
  { name: "squad", type: "span", category: "reading", n: 1000, d: -0.5 },
  { name: "triviaqa", type: "qa", category: "knowledge", n: 1000, d: 0.0 },
  { name: "naturalqs_open", type: "qa", category: "knowledge", n: 1000, d: 0.9 },
  { name: "popqa", type: "qa", category: "knowledge", n: 1000, d: 1.4 },
  { name: "bbh:cot", type: "bbh", category: "math", n: 1000, d: 0.6 },
  { name: "lambada:bpb", type: "bpb", category: "lm", n: 1000, d: 0.0 },
  { name: "alpaca_eval:v2", type: "judge", category: "instruct", n: 805, d: 0.8 },
  { name: "omniscience:judge", type: "omni", category: "knowledge", n: 600, d: 1.2 },
  { name: "tau2:retail", type: "agent", category: "agent", n: 115, d: 1.6 },
];

function buildTask(spec: TaskSpec): TaskDef {
  const base = spec.name.split(":")[0];
  let primary = "acc_per_char:default";
  let metrics: string[] = [];
  let kind: MetricKind = "binary";
  const metas: Record<string, MetricMeta> = {};
  let fewshot: number | null = 5;
  let maxTokens: number | null = null;
  let meanTokens = 0;
  let scale = 1;
  switch (spec.type) {
    case "mc":
      metrics = ["acc_per_char:default", "acc_raw:default", "acc_per_token:default"];
      metrics.forEach((m) => (metas[m] = meta("binary")));
      break;
    case "math":
    case "bbh":
      primary = "exact_match:flex";
      metrics = ["exact_match:flex", "exact_match:strict"];
      metrics.forEach((m) => (metas[m] = meta("binary")));
      fewshot = spec.type === "bbh" ? 3 : 8;
      maxTokens = spec.name === "aime24" ? 16384 : 1024;
      meanTokens = spec.name === "aime24" ? 4200 : spec.type === "bbh" ? 160 : 240;
      if (spec.name === "aime24") fewshot = 0;
      break;
    case "qa":
      primary = "exact_match:default";
      metrics = ["exact_match:default", "f1:default"];
      metas["exact_match:default"] = meta("binary");
      metas["f1:default"] = meta("bounded");
      maxTokens = 50;
      meanTokens = 8;
      break;
    case "span":
      primary = "f1:default";
      kind = "bounded";
      metrics = ["f1:default", "exact_match:default"];
      metas["f1:default"] = meta("bounded");
      metas["exact_match:default"] = meta("binary");
      fewshot = 3;
      maxTokens = 100;
      meanTokens = 12;
      break;
    case "code":
      primary = "pass_at_1:sandbox";
      metrics = ["pass_at_1:sandbox", "compile_rate:sandbox"];
      metas["pass_at_1:sandbox"] = meta("binary");
      metas["compile_rate:sandbox"] = meta("binary");
      fewshot = spec.name.startsWith("mbpp") ? 3 : 0;
      maxTokens = 2048;
      meanTokens = 260;
      break;
    case "safety":
      primary = "safe_response_rate:judge";
      metrics = ["safe_response_rate:judge", "over_refusal_rate:judge"];
      metas["safe_response_rate:judge"] = meta("binary");
      metas["over_refusal_rate:judge"] = meta("binary", false);
      fewshot = 0;
      maxTokens = 512;
      meanTokens = 140;
      break;
    case "ifeval":
      primary = "inst_level_loose_acc:ifeval";
      kind = "bounded";
      metrics = ["inst_level_loose_acc:ifeval", "prompt_level_loose_acc:ifeval", "prompt_level_strict_acc:ifeval"];
      metas["inst_level_loose_acc:ifeval"] = meta("bounded");
      metas["prompt_level_loose_acc:ifeval"] = meta("binary");
      metas["prompt_level_strict_acc:ifeval"] = meta("binary");
      fewshot = 0;
      maxTokens = 1280;
      meanTokens = 330;
      break;
    case "judge":
      primary = "win_rate:judge";
      kind = "bounded";
      metrics = ["win_rate:judge", "length_controlled_win_rate:judge"];
      metas["win_rate:judge"] = meta("bounded");
      metas["length_controlled_win_rate:judge"] = meta("bounded");
      fewshot = 0;
      maxTokens = 2048;
      meanTokens = 520;
      break;
    case "omni":
      primary = "omniscience_index:omniscience_judge";
      kind = "unbounded";
      metrics = [
        "omniscience_index:omniscience_judge",
        "accuracy:omniscience_judge",
        "hallucination_rate:omniscience_judge",
      ];
      metas["omniscience_index:omniscience_judge"] = meta("unbounded", true, "raw");
      metas["accuracy:omniscience_judge"] = meta("binary");
      metas["hallucination_rate:omniscience_judge"] = meta("binary", false);
      fewshot = 0;
      maxTokens = 256;
      meanTokens = 45;
      scale = 100;
      break;
    case "bpb":
      primary = "bits_per_byte:default";
      kind = "unbounded";
      metrics = ["bits_per_byte:default", "logits_per_char:default"];
      metas["bits_per_byte:default"] = meta("unbounded", false, "raw");
      metas["logits_per_char:default"] = meta("unbounded", true, "raw");
      fewshot = 0;
      break;
    case "agent":
      primary = "pass_hat_1:default";
      metrics = ["pass_hat_1:default"];
      metas["pass_hat_1:default"] = meta("binary");
      fewshot = 0;
      maxTokens = 4096;
      meanTokens = 1800;
      break;
  }
  return {
    name: spec.name,
    base,
    type: spec.type,
    category: spec.category,
    kind,
    primary,
    metrics,
    meta: metas,
    n: spec.n,
    difficulty: spec.d,
    spread: spec.spread ?? 1.1,
    chance: spec.chance ?? 0,
    fewshot,
    maxTokens,
    split: spec.type === "mc" && spec.category === "knowledge" ? "test" : "validation",
    scale,
    meanTokens,
    suite: spec.suite ?? null,
  };
}

export const TASKS: TaskDef[] = TASK_SPECS.map(buildTask);
export const TASK_BY_NAME = new Map(TASKS.map((t) => [t.name, t]));

// ---------------------------------------------------------------------------------------- suites

function suiteDef(name: string, aggregation: string, description: string, children: SuiteDef["children"]): SuiteDef {
  return { name, aggregation, description, children, definition_hash: hexId(16, "suite", name) };
}

const tasksOf = (suite: string) =>
  TASKS.filter((t) => t.suite === suite).map((t) => ({ type: "task" as const, name: t.name }));

export const SUITES: SuiteDef[] = [
  suiteDef("olmes:base", "average_of_averages", "OLMES base-model core: ARC, MMLU and commonsense.", [
    { type: "suite", name: "arc:rc" },
    { type: "suite", name: "mmlu:mc" },
    { type: "suite", name: "commonsense:rc" },
  ]),
  suiteDef("arc:rc", "average", "AI2 Reasoning Challenge, rank classification.", tasksOf("arc:rc")),
  suiteDef("mmlu:mc", "average", "MMLU multiple choice, macro average over subjects.", tasksOf("mmlu:mc")),
  suiteDef("commonsense:rc", "average", "Commonsense reasoning tasks, rank classification.", tasksOf("commonsense:rc")),
  suiteDef("math:olmo3", "average", "Math word problems and competition math.", tasksOf("math:olmo3")),
  suiteDef("code:olmo3", "average", "Code generation with sandboxed execution.", tasksOf("code:olmo3")),
  suiteDef("safety:olmo3", "average", "Refusal and over-refusal safety evaluations.", tasksOf("safety:olmo3")),
];
export const SUITE_BY_NAME = new Map(SUITES.map((s) => [s.name, s]));

export function suiteLeaves(name: string): string[] {
  const def = SUITE_BY_NAME.get(name);
  if (!def) return [];
  return def.children.flatMap((c) => (c.type === "task" ? [c.name] : suiteLeaves(c.name)));
}

/** Suites (outermost first) that cover any of the given tasks. */
export function suitesFor(taskNames: Set<string>): SuiteDef[] {
  return SUITES.filter((s) => suiteLeaves(s.name).some((t) => taskNames.has(t)));
}

export function parentSuite(name: string): string | null {
  for (const s of SUITES) if (s.children.some((c) => c.type === "suite" && c.name === name)) return s.name;
  return null;
}

// ---------------------------------------------------------------------------------------- models

type Ability = Partial<Record<Category, [number, number]>>;

interface SeriesSpec {
  series: string;
  family: string;
  author: string;
  group: string | null;
  steps: (number | null)[];
  ability: Ability;
  tasks: string[];
  pathFor: (step: number | null) => string;
  revisionFor?: (step: number | null) => string | null;
  size: number;
  gpuType: string;
  gpuCount: number;
  thinking?: boolean;
  daysAgoStart: number;
  daysAgoEnd: number;
  tokensPerStep?: number;
}

const ALL = TASKS.map((t) => t.name);
const BASE_SET = TASKS.filter(
  (t) =>
    t.suite === "olmes:base" ||
    ["arc:rc", "mmlu:mc", "commonsense:rc", "math:olmo3", "code:olmo3"].includes(t.suite ?? "") ||
    ["drop", "squad", "triviaqa", "naturalqs_open", "popqa", "bbh:cot", "lambada:bpb"].includes(t.name),
).map((t) => t.name);
const POST_SET = TASKS.filter(
  (t) =>
    ["arc:rc", "mmlu:mc", "commonsense:rc", "math:olmo3", "code:olmo3", "safety:olmo3"].includes(t.suite ?? "") ||
    ["ifeval", "alpaca_eval:v2", "bbh:cot", "omniscience:judge", "tau2:retail", "popqa"].includes(t.name),
).map((t) => t.name);

const range = (start: number, stop: number, step: number) => {
  const out: number[] = [];
  for (let v = start; v <= stop; v += step) out.push(v);
  return out;
};

export const SERIES: SeriesSpec[] = [
  {
    series: "olmo3-7b-midtrain",
    family: "olmo3",
    author: "chrisg",
    group: "olmo3-7b-midtrain-ablations",
    steps: range(1000, 10000, 1000),
    ability: {
      knowledge: [0.35, 0.55],
      commonsense: [0.9, 0.35],
      math: [-0.4, 1.4],
      code: [-0.6, 1.1],
      reading: [0.4, 0.5],
      lm: [0.4, 0.4],
    },
    tasks: BASE_SET,
    pathFor: (s) => `gs://ai2-llm/checkpoints/olmo3-7b-midtrain/step${s}-hf`,
    revisionFor: (s) => `step${s}`,
    size: 7,
    gpuType: "NVIDIA H100 80GB HBM3",
    gpuCount: 1,
    daysAgoStart: 13,
    daysAgoEnd: 0.1,
    tokensPerStep: 4_194_304,
  },
  {
    series: "olmo3-1b-anneal",
    family: "olmo3",
    author: "ana",
    group: "olmo3-1b-anneal-sweep",
    steps: range(500, 4000, 500),
    ability: {
      knowledge: [-0.5, 0.45],
      commonsense: [0.2, 0.35],
      math: [-1.6, 1.0],
      code: [-1.9, 0.7],
      reading: [-0.4, 0.45],
      lm: [-0.4, 0.5],
    },
    tasks: BASE_SET,
    pathFor: (s) => `gs://ai2-llm/checkpoints/olmo3-1b-anneal/step${s}-hf`,
    revisionFor: (s) => `step${s}`,
    size: 1,
    gpuType: "NVIDIA A100-SXM4-80GB",
    gpuCount: 1,
    daysAgoStart: 20,
    daysAgoEnd: 4,
    tokensPerStep: 2_097_152,
  },
  {
    series: "olmo3-32b-sft",
    family: "olmo3",
    author: "raj",
    group: "olmo3-32b-sft-sweep",
    steps: range(400, 2400, 400),
    ability: {
      knowledge: [1.3, 0.2],
      commonsense: [1.5, 0.15],
      math: [1.0, 0.7],
      code: [0.6, 0.7],
      safety: [0.4, 1.0],
      instruct: [0.0, 1.2],
      agent: [-0.4, 0.9],
    },
    tasks: POST_SET,
    pathFor: (s) => `gs://ai2-llm/checkpoints/olmo3-32b-sft/step${s}`,
    revisionFor: (s) => `step${s}`,
    size: 32,
    gpuType: "NVIDIA H100 80GB HBM3",
    gpuCount: 4,
    daysAgoStart: 9,
    daysAgoEnd: 1,
  },
  {
    series: "tulu4-8b-dpo",
    family: "tulu4",
    author: "maliam",
    group: "tulu4-dpo-safety",
    steps: range(250, 1750, 250),
    ability: {
      knowledge: [0.6, 0.1],
      commonsense: [1.0, 0.05],
      math: [0.4, 0.3],
      code: [0.1, 0.2],
      safety: [0.5, 1.4],
      instruct: [0.3, 0.9],
      agent: [-0.9, 0.5],
    },
    tasks: POST_SET,
    pathFor: (s) => `gs://ai2-llm/checkpoints/tulu4-8b-dpo/step${s}`,
    revisionFor: (s) => `step${s}`,
    size: 8,
    gpuType: "NVIDIA H100 80GB HBM3",
    gpuCount: 1,
    daysAgoStart: 6,
    daysAgoEnd: 0.4,
  },
];

interface RefSpec {
  name: string;
  family: string;
  ability: Ability;
  size: number;
  thinking?: boolean;
}

export const REFERENCES: RefSpec[] = [
  {
    name: "Qwen/Qwen3-8B",
    family: "qwen3",
    size: 8,
    thinking: true,
    ability: {
      knowledge: [1.25, 0],
      commonsense: [1.2, 0],
      math: [2.0, 0],
      code: [1.3, 0],
      safety: [1.1, 0],
      instruct: [1.0, 0],
      reading: [1.0, 0],
      lm: [0.6, 0],
      agent: [0.7, 0],
    },
  },
  {
    name: "meta-llama/Llama-3.1-8B",
    family: "llama3",
    size: 8,
    ability: {
      knowledge: [0.9, 0],
      commonsense: [1.25, 0],
      math: [0.2, 0],
      code: [0.0, 0],
      safety: [0.6, 0],
      instruct: [0.4, 0],
      reading: [0.8, 0],
      lm: [0.9, 0],
      agent: [-0.6, 0],
    },
  },
  {
    name: "allenai/OLMo-2-1124-7B",
    family: "olmo2",
    size: 7,
    ability: {
      knowledge: [0.75, 0],
      commonsense: [1.15, 0],
      math: [0.3, 0],
      code: [-0.4, 0],
      safety: [0.2, 0],
      instruct: [0.0, 0],
      reading: [0.7, 0],
      lm: [0.7, 0],
      agent: [-1.2, 0],
    },
  },
];

// ---------------------------------------------------------------------------------------- runs

export interface MockTaskResult {
  id: number;
  runId: string;
  task: TaskDef;
  taskHash: string;
  n: number;
  limit: number | null;
  ability: number;
  noiseKey: string;
  runKey: string;
  error: string | null;
  /** Seconds from the run's processing start until this task finished. */
  durationS: number;
  runtime: MockTaskRuntime;
  createdAt: string;
  thinking: boolean;
  maxTokens: number | null;
}

export interface MockTaskRuntime {
  inferenceS: number | null;
  basis: RuntimeBasis;
  spanStart: number | null;
  spanEnd: number | null;
  promptTokens: number;
  completionTokens: number;
  tokenShare: number | null;
}

export interface MockRun {
  summary: RunSummary;
  /** Model load and server start; null for runs that predate runtime recording. */
  startupS: number | null;
  processingStartedAt: string | null;
  processingS: number | null;
  model: ModelDetail;
  beaker: BeakerDetail | null;
  environment: EnvironmentDetail;
  git: { repo: string; commit: string; branch: string; dirty: boolean };
  argv: string[];
  taskSpecs: string[];
  notes: string | null;
  errors: { task: string | null; error: string }[];
  taskResultIds: number[];
  thinking: boolean;
  vllmVersion: string;
  maxModelLen: number;
  updatedAt: string;
}

const USERS = ["chrisg", "ana", "raj", "maliam", "kylel"];
export const ME = "chrisg";

function iso(t: number): string {
  return new Date(t).toISOString().replace(".000Z", "+00:00");
}

function abilityFor(spec: Ability, cat: Category, frac: number): number {
  const entry = spec[cat] ?? [0, 0];
  return entry[0] + entry[1] * frac;
}

interface RunPlan {
  modelName: string;
  series: string;
  family: string;
  step: number | null;
  tokensSeen: number | null;
  path: string;
  revision: string | null;
  author: string;
  group: string | null;
  launch: string | null;
  createdAt: number;
  tasks: string[];
  ability: (task: TaskDef) => number;
  noiseKey: string;
  status: RunStatus;
  uploadState: UploadState;
  failTasks: Record<string, string>;
  size: number;
  gpuType: string;
  gpuCount: number;
  thinking: boolean;
  tags: string[];
  limits: Record<string, number>;
  variant: string;
  maxModelLen: number;
  workspace: string;
  priority: string;
  doneFraction?: number;
  notes?: string;
  branch?: string;
}

function makePlans(): RunPlan[] {
  const plans: RunPlan[] = [];
  for (const s of SERIES) {
    const steps = s.steps as number[];
    const maxStep = steps[steps.length - 1];
    steps.forEach((step, idx) => {
      const frac = Math.log1p(step / (maxStep / 10)) / Math.log1p(10);
      const createdAt =
        NOW - (s.daysAgoStart - ((s.daysAgoStart - s.daysAgoEnd) * idx) / Math.max(1, steps.length - 1)) * DAY;
      const affinityKey = s.series;
      const plan: RunPlan = {
        modelName: `${s.series}-step${step}`,
        series: s.series,
        family: s.family,
        step,
        tokensSeen: s.tokensPerStep ? step * s.tokensPerStep : null,
        path: s.pathFor(step),
        revision: s.revisionFor?.(step) ?? null,
        author: s.author,
        group: s.group,
        launch: `L-${hexId(8, s.series, Math.floor(idx / 2))}`,
        createdAt,
        tasks: s.tasks,
        ability: (task) => {
          const rng = rngFor("affinity", affinityKey, task.name);
          const affinity = normal(rng) * 0.18;
          const jitter = normal(rngFor("jit", s.series, step, task.name)) * 0.05;
          return abilityFor(s.ability, task.category, frac) + affinity + jitter;
        },
        noiseKey: s.series,
        status: "complete",
        uploadState: "complete",
        failTasks: {},
        size: s.size,
        gpuType: s.gpuType,
        gpuCount: s.gpuCount,
        thinking: false,
        tags: [],
        limits: s.series === "olmo3-1b-anneal" ? { "gsm8k:cot:olmo3": 500 } : {},
        variant: "a",
        maxModelLen: 4096,
        workspace: s.series.startsWith("tulu") ? "ai2/tulu" : "ai2/olmo-eval",
        priority: idx === steps.length - 1 ? "urgent" : "high",
      };
      plans.push(plan);
    });
  }
  // Mark a few partial runs and decorate.
  const find = (name: string) => plans.find((p) => p.modelName === name)!;
  find("olmo3-32b-sft-step1600").failTasks = {
    "livecodebench:v6": "SandboxTimeoutError: 41 of 300 problems timed out after 60s in the sandbox",
  };
  find("olmo3-32b-sft-step1600").status = "partial";
  const p10k = find("olmo3-7b-midtrain-step10000");
  p10k.failTasks = {
    "humaneval:pass@1": "RuntimeError: sandbox pool exhausted (0 of 16 workers healthy)",
    "mbpp:pass@1": "RuntimeError: sandbox pool exhausted (0 of 16 workers healthy)",
  };
  p10k.status = "partial";
  p10k.tags = ["midtrain", "candidate"];
  find("olmo3-7b-midtrain-step9000").tags = ["midtrain", "candidate"];
  find("olmo3-7b-midtrain-step5000").tags = ["midtrain"];
  find("tulu4-8b-dpo-step1500").tags = ["release-candidate"];
  find("olmo3-1b-anneal-step500").tags = ["debug"];
  find("olmo3-7b-midtrain-step8000").notes = "Learning rate warmup changed at step 7500; watch MMLU.";

  // Rerun of the failed code tasks for the 7B step 10000 checkpoint.
  plans.push({
    ...p10k,
    launch: `L-${hexId(8, "rerun-10k")}`,
    createdAt: p10k.createdAt + 3 * HOUR,
    tasks: ["humaneval:pass@1", "mbpp:pass@1"],
    failTasks: {},
    status: "complete",
    tags: ["rerun"],
    variant: "rerun",
  });
  // A second settings variant (longer context) for step 8000.
  const p8k = find("olmo3-7b-midtrain-step8000");
  plans.push({
    ...p8k,
    launch: `L-${hexId(8, "ctx8k")}`,
    createdAt: p8k.createdAt + 5 * HOUR,
    maxModelLen: 8192,
    variant: "ctx8k",
    tags: ["long-context"],
    notes: undefined,
  });
  // A run in progress (fresh) and a stale one.
  const s7 = SERIES[0];
  plans.push({
    ...p10k,
    modelName: "olmo3-7b-midtrain-step11000",
    step: 11000,
    tokensSeen: 11000 * (s7.tokensPerStep ?? 0),
    path: s7.pathFor(11000),
    revision: "step11000",
    launch: `L-${hexId(8, "11k")}`,
    createdAt: NOW - 1.4 * HOUR,
    status: "running",
    failTasks: {},
    tags: [],
    variant: "a",
    doneFraction: 0.45,
    ability: (task) => p10k.ability(task) + 0.03,
  });
  const s1 = SERIES[1];
  const p1last = find("olmo3-1b-anneal-step4000");
  plans.push({
    ...p1last,
    modelName: "olmo3-1b-anneal-step4500",
    step: 4500,
    tokensSeen: 4500 * (s1.tokensPerStep ?? 0),
    path: s1.pathFor(4500),
    revision: "step4500",
    launch: `L-${hexId(8, "1b-4500")}`,
    createdAt: NOW - 3.2 * DAY,
    status: "running",
    doneFraction: 0.2,
    ability: (task) => p1last.ability(task) + 0.02,
  });
  // Uploading and failed runs.
  const tlast = find("tulu4-8b-dpo-step1750");
  plans.push({
    ...tlast,
    modelName: "tulu4-8b-dpo-step2000",
    step: 2000,
    path: SERIES[3].pathFor(2000),
    revision: "step2000",
    launch: `L-${hexId(8, "tulu-2000")}`,
    createdAt: NOW - 0.6 * HOUR,
    uploadState: "uploading",
    ability: (task) => tlast.ability(task) + 0.02,
  });
  const s32 = find("olmo3-32b-sft-step2400");
  plans.push({
    ...s32,
    modelName: "olmo3-32b-sft-step2800",
    step: 2800,
    path: SERIES[2].pathFor(2800),
    revision: "step2800",
    launch: `L-${hexId(8, "32b-2800")}`,
    createdAt: NOW - 7 * HOUR,
    status: "failed",
    tasks: [],
    failTasks: {
      "*": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB (GPU 3; 79.10 GiB total)",
    },
  });

  // Reference models.
  const A100 = "NVIDIA A100-SXM4-80GB";
  // The last entries run a single task, so their task runtime is measured directly.
  const refRuns: [RefSpec, string, string | null, number, { gpu?: string; tasks?: string[] }?][] = [
    [REFERENCES[1], "chrisg", "olmo3-7b-midtrain-ablations", 12.5],
    [REFERENCES[2], "chrisg", "olmo3-7b-midtrain-ablations", 12.4],
    [REFERENCES[0], "maliam", "tulu4-dpo-safety", 5.9],
    [REFERENCES[0], "kylel", null, 2.2, { gpu: A100 }],
    [REFERENCES[1], "raj", "olmo3-32b-sft-sweep", 8.8],
    [REFERENCES[2], "kylel", null, 1.6, { gpu: A100, tasks: ["gsm8k:cot:olmo3"] }],
    [REFERENCES[1], "kylel", null, 0.9, { gpu: A100, tasks: ["gsm8k:cot:olmo3"] }],
    [REFERENCES[2], "chrisg", "olmo3-7b-midtrain-ablations", 0.7, { tasks: ["ifeval"] }],
  ];
  refRuns.forEach(([ref, author, group, daysAgo, extra], i) => {
    plans.push({
      modelName: ref.name,
      series: ref.name,
      family: ref.family,
      step: null,
      tokensSeen: null,
      path: ref.name,
      revision: "main",
      author,
      group,
      launch: `L-${hexId(8, "ref", i)}`,
      createdAt: NOW - daysAgo * DAY,
      tasks: extra?.tasks ?? (ref.name.includes("Llama") && group?.includes("7b") ? BASE_SET : ref.name.includes("OLMo-2") ? BASE_SET : ALL),
      ability: (task) => {
        const affinity = normal(rngFor("affinity", ref.name, task.name)) * 0.2;
        return abilityFor(ref.ability, task.category, 1) + affinity;
      },
      noiseKey: ref.name,
      status: "complete",
      uploadState: "complete",
      failTasks: {},
      size: ref.size,
      gpuType: extra?.gpu ?? "NVIDIA H100 80GB HBM3",
      gpuCount: 1,
      thinking: !!ref.thinking,
      tags: ["reference"],
      limits: {},
      variant: `ref${i}`,
      maxModelLen: ref.thinking ? 32768 : 8192,
      workspace: "ai2/olmo-eval",
      priority: "normal",
      branch: author === "kylel" ? "kylel/qwen3-thinking" : undefined,
    });
  });
  return plans;
}

// ---------------------------------------------------------------------------------------- runtime

/** Prompt tokens per instance; the inference tab uses the same estimate. */
export function promptTokensPerInstance(task: TaskDef): number {
  return 180 + (task.fewshot ?? 0) * 140;
}

/**
 * Processor sharing: up to `concurrency` tasks share the inference workers, each progressing at
 * an equal share of their throughput. `work` is each task's time with the workers to itself.
 */
export function shareWorkers(work: number[], concurrency: number): { start: number; end: number }[] {
  const out = work.map(() => ({ start: 0, end: 0 }));
  const rem = [...work];
  const active: number[] = [];
  let next = 0;
  let t = 0;
  while (next < work.length && active.length < concurrency) active.push(next++);
  while (active.length) {
    const step = Math.min(...active.map((i) => rem[i]));
    t += step * active.length;
    for (const i of active) rem[i] -= step;
    for (const i of active.filter((j) => rem[j] <= 1e-9)) {
      out[i].end = t;
      active.splice(active.indexOf(i), 1);
    }
    while (next < work.length && active.length < concurrency) {
      out[next].start = t;
      active.push(next++);
    }
  }
  return out;
}

interface RunSim {
  basis: RuntimeBasis;
  startupS: number | null;
  processingS: number;
  tasks: { end: number; runtime: MockTaskRuntime }[];
}

/**
 * Runtime of one run: startup, then the tasks sharing inference workers. Runs older than two
 * weeks predate runtime recording; single-task runs are measured; recent runs have per-batch
 * metrics (attributed); the rest are estimated from the task's token share.
 */
function simulateRun(plan: RunPlan, runId: string, taskNames: string[]): RunSim {
  const a100 = plan.gpuType.includes("A100");
  const sizeSpeed = plan.size >= 32 ? 0.7 : plan.size >= 7 ? 1 : 3;
  const speed = sizeSpeed * (a100 ? 0.55 : 1);
  const promptRate = 25_000 * speed;
  const genRate = 2_200 * speed;
  const tokens = taskNames.map((name) => {
    const task = TASK_BY_NAME.get(name)!;
    const n = Math.min(task.n, plan.limits[name] ?? task.n);
    return {
      prompt: n * promptTokensPerInstance(task),
      completion: Math.round(n * task.meanTokens * (plan.thinking ? 4 : 1)),
    };
  });
  const work = tokens.map(
    (tk, k) => (12 + tk.prompt / promptRate + tk.completion / genRate) * (0.85 + 0.3 * rngFor("work", runId, taskNames[k])()),
  );
  const spans = shareWorkers(work, 6);
  const processingS = spans.reduce((m, sp) => Math.max(m, sp.end), 0);
  const startRng = rngFor("startup", runId);
  const startupBase = plan.size >= 32 ? 460 : plan.size >= 7 ? 190 : 110;
  const startupS = Math.round(startupBase * (0.9 + 0.25 * startRng()) * (a100 ? 1.15 : 1) * 10) / 10;
  const old = plan.createdAt < NOW - 14 * DAY;
  const basis: RuntimeBasis = old
    ? "not_recorded"
    : taskNames.length === 1
      ? "measured"
      : plan.status === "running" || plan.createdAt > NOW - 2.5 * DAY
        ? "attributed"
        : "estimated";
  const totalTokens = tokens.reduce((acc, tk) => acc + tk.prompt + tk.completion, 0);
  const tasks = taskNames.map((name, k) => {
    const share = totalTokens ? (tokens[k].prompt + tokens[k].completion) / totalTokens : null;
    let inferenceS: number | null = null;
    if (basis === "measured") inferenceS = processingS;
    else if (basis === "attributed") inferenceS = work[k] * (0.97 + 0.06 * rngFor("attr", runId, name)());
    else if (basis === "estimated" && share != null) inferenceS = processingS * share;
    return {
      end: spans[k].end,
      runtime: {
        inferenceS: inferenceS != null ? Math.round(inferenceS * 10) / 10 : null,
        basis,
        spanStart: old ? null : Math.round(spans[k].start * 10) / 10,
        spanEnd: old ? null : Math.round(spans[k].end * 10) / 10,
        promptTokens: tokens[k].prompt,
        completionTokens: tokens[k].completion,
        tokenShare: old ? null : share,
      },
    };
  });
  return { basis, startupS: old ? null : startupS, processingS, tasks };
}

// ---------------------------------------------------------------------------------------- world

export interface World {
  runs: MockRun[];
  runById: Map<string, MockRun>;
  taskResults: Map<number, MockTaskResult>;
  models: Map<string, ModelDetail>;
  users: string[];
}

let world: World | null = null;

export function getWorld(): World {
  if (!world) world = buildWorld();
  return world;
}

function buildWorld(): World {
  const runs: MockRun[] = [];
  const runById = new Map<string, MockRun>();
  const taskResults = new Map<number, MockTaskResult>();
  const models = new Map<string, ModelDetail>();
  let trId = 1000;

  for (const plan of makePlans()) {
    const runId = hexId(12, "run", plan.modelName, plan.variant);
    const settingsHash = hexId(16, "settings", plan.series, plan.maxModelLen);
    const modelHash = hexId(16, "mhash", plan.path, plan.maxModelLen);
    const modelId = hexId(12, "mid", plan.modelName, modelHash);
    const seriesLabel = plan.series.includes("/") ? plan.series.split("/").pop()! : plan.series;
    const vllmVersion = plan.createdAt > NOW - 7 * DAY ? "0.11.0" : "0.10.2";
    const providerConfig = {
      kind: "vllm",
      model: plan.path,
      revision: plan.revision,
      max_model_len: plan.maxModelLen,
      tensor_parallel_size: plan.gpuCount,
      gpu_memory_utilization: 0.9,
      dtype: "bfloat16",
      enable_prefix_caching: true,
      max_num_seqs: plan.size >= 32 ? 128 : 256,
      chat_template: plan.series.includes("sft") || plan.series.includes("dpo") || plan.thinking ? "tokenizer" : null,
      generation_defaults: { temperature: 0.0, top_p: 1.0, seed: 1234 },
    };
    const model: ModelDetail = {
      model_id: modelId,
      name: plan.modelName,
      model_hash: modelHash,
      series: plan.series,
      series_label: seriesLabel,
      family: plan.family,
      step: plan.step,
      tokens_seen: plan.tokensSeen,
      path: plan.path,
      revision: plan.revision,
      provider_kind: "vllm",
      provider_config: providerConfig,
      settings_hash: settingsHash,
    };
    models.set(modelId, model);

    const failedRun = plan.status === "failed";
    const doneCount = plan.doneFraction != null ? Math.round(plan.tasks.length * plan.doneFraction) : plan.tasks.length;
    const taskNames = plan.tasks.slice(0, doneCount);
    const trIds: number[] = [];
    const started = plan.createdAt - 30_000;
    const sim = simulateRun(plan, runId, taskNames);
    const processingStart = started + (sim.startupS ?? 0) * 1000;
    taskNames.forEach((name, k) => {
      const task = TASK_BY_NAME.get(name)!;
      const limit = plan.limits[name] ?? null;
      const n = Math.min(task.n, limit ?? task.n);
      const tr: MockTaskResult = {
        id: trId++,
        runId,
        task,
        taskHash: hexId(16, "taskhash", name, limit ?? "full"),
        n,
        limit,
        ability: plan.ability(task),
        noiseKey: plan.noiseKey,
        runKey: runId,
        error: plan.failTasks[name] ?? null,
        durationS: Math.round(sim.tasks[k].end),
        runtime: sim.tasks[k].runtime,
        createdAt: iso(processingStart + sim.tasks[k].end * 1000),
        thinking: plan.thinking,
        maxTokens: task.maxTokens && plan.thinking ? Math.max(task.maxTokens, 8192) : task.maxTokens,
      };
      taskResults.set(tr.id, tr);
      trIds.push(tr.id);
    });
    const lastTaskEnd = processingStart + sim.processingS * 1000;
    const status = plan.status;
    const finished = status === "running" ? null : failedRun ? plan.createdAt + 14 * 60_000 : lastTaskEnd + 2 * 60_000;
    const duration = finished ? (finished - started) / 1000 : null;
    const recorded = sim.basis !== "not_recorded";
    const experimentId = beakerId("ex", runId);
    const datasetId = beakerId("ds", runId);
    const commit = hexId(40, "commit", Math.floor((NOW - plan.createdAt) / (3 * DAY)));
    const branch = plan.branch ?? "main";
    const gcsPrefix = `gs://${RESULTS_BUCKET}/runs/${runId}/`;
    const links: RunLinks = {
      beaker_experiment: `https://beaker.org/ex/${experimentId}`,
      beaker_result_dataset: `https://beaker.org/ds/${datasetId}`,
      beaker_workspace: `https://beaker.org/ws/${plan.workspace}`,
      gcs_console: `https://console.cloud.google.com/storage/browser/${RESULTS_BUCKET}/runs/${runId}?project=ai2-skiff2-olmo-eval`,
      github_commit: `https://github.com/allenai/olmo-eval/commit/${commit}`,
    };
    const errors = Object.entries(plan.failTasks).map(([task, error]) => ({
      task: task === "*" ? null : task,
      error,
    }));
    const nFailed = errors.filter((e) => e.task).length;
    const numInstances = trIds.reduce((acc, id) => acc + (taskResults.get(id)!.error ? 0 : taskResults.get(id)!.n), 0);
    const summary: RunSummary = {
      run_id: runId,
      launch_id: plan.launch,
      experiment_name: `${seriesLabel.toLowerCase().replace(/[^a-z0-9]+/g, "-")}${plan.step != null ? `-step${plan.step}` : ""}-eval`,
      experiment_group: plan.group,
      status,
      upload_state: plan.uploadState,
      stale: status === "running" && NOW - plan.createdAt > DAY,
      model: {
        model_id: model.model_id,
        name: model.name,
        model_hash: model.model_hash,
        series: model.series,
        series_label: model.series_label,
        family: model.family,
        step: model.step,
        tokens_seen: model.tokens_seen,
      },
      author: plan.author,
      uploaded_by: `${plan.author}@allenai.org`,
      tags: plan.tags,
      created_at: iso(plan.createdAt),
      started_at: iso(started),
      finished_at: finished ? iso(finished) : null,
      duration_seconds: duration,
      num_tasks: trIds.length,
      num_failed_tasks: nFailed,
      num_instances: numInstances,
      headline: null,
      beaker_experiment_id: experimentId,
      beaker_workspace: plan.workspace,
      git_commit: commit,
      git_branch: branch,
      gcs_prefix: gcsPrefix,
      links,
    };
    const run: MockRun = {
      summary,
      startupS: sim.startupS,
      processingStartedAt: recorded && sim.startupS != null && !failedRun ? iso(processingStart) : null,
      processingS: recorded && status !== "running" && !failedRun ? sim.processingS : null,
      model,
      beaker: {
        experiment_id: experimentId,
        workload_id: beakerId("wl", runId),
        job_id: beakerId("job", runId),
        task_id: beakerId("task", runId),
        workspace: plan.workspace,
        workspace_id: beakerId("ws", plan.workspace),
        result_dataset_id: datasetId,
        node_hostname: `jupiter-cs-aus-${100 + (hashString(runId) % 300)}.reviz.ai2.in`,
        cluster: plan.size >= 32 ? "ai2/jupiter" : hashString(runId) % 3 === 0 ? "ai2/saturn" : "ai2/jupiter",
        priority: plan.priority,
        image: `beaker://${plan.author}/olmo-eval-${commit.slice(0, 7)}`,
        budget: "ai2/oe-eval",
        gpu_count: plan.gpuCount,
        cpu_count: plan.gpuCount * 16,
      },
      environment: {
        hostname: `jupiter-cs-aus-${100 + (hashString(runId) % 300)}`,
        platform: "Linux-6.8.0-1028-gcp-x86_64-with-glibc2.35",
        python_version: "3.12.11",
        packages: {
          "olmo-eval": "0.9.3",
          vllm: vllmVersion,
          torch: "2.8.0",
          transformers: "4.56.1",
          numpy: "2.2.6",
          "flash-attn": "2.8.3",
        },
        cuda_version: "12.8",
        gpu_type: plan.gpuType,
        gpu_count: plan.gpuCount,
      },
      git: { repo: "allenai/olmo-eval", commit, branch, dirty: plan.author === "ana" && plan.step === 2500 },
      argv: [
        "olmo-eval",
        "beaker",
        "launch",
        "-m",
        plan.path,
        ...(plan.revision && !plan.path.endsWith(plan.revision) ? ["--revision", plan.revision] : []),
        ...suitesFor(new Set(plan.tasks))
          .filter((s) => !parentSuite(s.name))
          .flatMap((s) => ["-t", s.name]),
        ...plan.tasks.filter((name) => !TASK_BY_NAME.get(name)!.suite).flatMap((name) => ["-t", name]),
        "--gpus",
        String(plan.gpuCount),
        "--cluster",
        "ai2/jupiter",
        "--workspace",
        plan.workspace,
        ...(plan.group ? ["--experiment-group", plan.group] : []),
        "--priority",
        plan.priority,
      ],
      taskSpecs: [
        ...suitesFor(new Set(plan.tasks))
          .filter((s) => !parentSuite(s.name))
          .map((s) => s.name),
        ...plan.tasks.filter((name) => !TASK_BY_NAME.get(name)!.suite),
      ],
      notes: plan.notes ?? null,
      errors,
      taskResultIds: trIds,
      thinking: plan.thinking,
      vllmVersion,
      maxModelLen: plan.maxModelLen,
      updatedAt: iso(finished ?? NOW - 20 * 60_000),
    };
    runs.push(run);
    runById.set(runId, run);
  }
  runs.sort((a, b) => b.summary.created_at.localeCompare(a.summary.created_at));
  const w: World = { runs, runById, taskResults, models, users: USERS };
  world = w;
  for (const run of runs) run.summary.headline = computeHeadline(run);
  return w;
}

// ---------------------------------------------------------------------------------------- instances

const difficultyCache = new Map<string, Float64Array>();
function difficulties(task: TaskDef): Float64Array {
  let d = difficultyCache.get(task.name);
  if (!d) {
    const rng = rngFor("difficulty", task.name);
    d = new Float64Array(task.n);
    for (let i = 0; i < task.n; i++) d[i] = task.difficulty + normal(rng) * task.spread;
    difficultyCache.set(task.name, d);
  }
  return d;
}

const noiseCache = new Map<string, Float64Array>();
function noise(key: string, n: number): Float64Array {
  let arr = noiseCache.get(key);
  if (!arr || arr.length < n) {
    const rng = rngFor("noise", key);
    arr = new Float64Array(n);
    for (let i = 0; i < n; i++) arr[i] = normal(rng);
    noiseCache.set(key, arr);
  }
  return arr;
}

/** Standard normal CDF (Abramowitz-Stegun). */
export function phi(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}

export interface InstanceData {
  primary: Float64Array;
  /** Uniform draw per instance used for secondary attributes. */
  u: Float64Array;
  /** Model success probability per instance. */
  p: Float64Array;
}

const instanceCache = new Map<number, InstanceData>();

export function instanceData(tr: MockTaskResult): InstanceData {
  const cached = instanceCache.get(tr.id);
  if (cached) return cached;
  const task = tr.task;
  const d = difficulties(task);
  const zs = noise(`${tr.noiseKey}|${task.name}`, task.n);
  const zr = noise(`${tr.runKey}|${task.name}`, task.n);
  const primary = new Float64Array(tr.n);
  const uArr = new Float64Array(tr.n);
  const pArr = new Float64Array(tr.n);
  for (let i = 0; i < tr.n; i++) {
    const p = task.chance + (1 - task.chance) * sigmoid(1.7 * (tr.ability - d[i]));
    const u = phi(0.88 * zs[i] + 0.475 * zr[i]);
    pArr[i] = p;
    uArr[i] = u;
    let score: number;
    switch (task.type) {
      case "span":
      case "ifeval":
      case "judge":
        score = clamp(Math.round((p + (0.5 - u) * 0.9) * 20) / 20, 0, 1);
        break;
      case "bpb":
        score = 0.95 + 0.22 * d[i] - 0.2 * tr.ability + 0.18 * (u - 0.5);
        break;
      case "omni":
        score = u < p * 0.85 ? 1 : u < p * 0.85 + 0.35 * (1 - p) ? -1 : 0;
        break;
      default:
        score = u < p ? 1 : 0;
    }
    primary[i] = score;
  }
  const data = { primary, u: uArr, p: pArr };
  instanceCache.set(tr.id, data);
  return data;
}

export function nativeId(task: TaskDef, i: number): string {
  const prefix = task.base.replace(/[^a-z0-9]+/gi, "_");
  return `${prefix}_${task.split}_${String(i).padStart(4, "0")}`;
}

export function indexOfNative(_task: TaskDef, native: string): number {
  const m = /_(\d+)$/.exec(native);
  return m ? Number(m[1]) : -1;
}

/** Secondary metric for an instance, derived from the primary deterministically. */
export function instanceMetrics(tr: MockTaskResult, i: number): Record<string, number | null> {
  const data = instanceData(tr);
  const v = data.primary[i];
  const u = data.u[i];
  const out: Record<string, number | null> = {};
  for (const key of tr.task.metrics) {
    if (key === tr.task.primary) {
      out[key] = v;
      continue;
    }
    const flip = ((u * 9973) % 1) < 0.08;
    switch (key) {
      case "exact_match:strict":
        out[key] = v === 1 && flip ? 0 : v;
        break;
      case "acc_raw:default":
      case "acc_per_token:default":
        out[key] = flip ? 1 - v : v;
        break;
      case "f1:default":
        out[key] = v === 1 ? 1 : Math.round(u * 0.6 * 100) / 100;
        break;
      case "exact_match:default":
        out[key] = v >= 0.95 ? 1 : 0;
        break;
      case "compile_rate:sandbox":
        out[key] = v === 1 || u < 0.9 ? 1 : 0;
        break;
      case "over_refusal_rate:judge":
        out[key] = v === 1 && u < 0.12 ? 1 : 0;
        break;
      case "prompt_level_loose_acc:ifeval":
        out[key] = v >= 0.99 ? 1 : 0;
        break;
      case "prompt_level_strict_acc:ifeval":
        out[key] = v >= 0.99 && !flip ? 1 : 0;
        break;
      case "length_controlled_win_rate:judge":
        out[key] = clamp(v + (u - 0.5) * 0.1, 0, 1);
        break;
      case "accuracy:omniscience_judge":
        out[key] = v === 1 ? 1 : 0;
        break;
      case "hallucination_rate:omniscience_judge":
        out[key] = v === -1 ? 1 : 0;
        break;
      case "logits_per_char:default":
        out[key] = -v * 0.69;
        break;
      default:
        out[key] = v;
    }
  }
  return out;
}

export function completionTokens(tr: MockTaskResult, i: number): number | null {
  const task = tr.task;
  if (!task.meanTokens) return null;
  const data = instanceData(tr);
  const z = normal(rngFor("len", tr.runKey, task.name, i));
  const correctBoost = data.primary[i] >= 0.5 ? 0.9 : 1.25;
  const mean = task.meanTokens * (tr.thinking ? 4 : 1) * correctBoost;
  const value = Math.round(mean * Math.exp(0.45 * z));
  const cap = tr.maxTokens ?? 100000;
  // Long tails hit the limit.
  const runaway = data.primary[i] < 0.5 && data.u[i] > 0.93 && task.type !== "qa";
  return runaway ? cap : Math.max(3, Math.min(cap, value));
}

export function finishReason(tr: MockTaskResult, i: number): string | null {
  const tokens = completionTokens(tr, i);
  if (tokens == null) return null;
  return tr.maxTokens != null && tokens >= tr.maxTokens ? "length" : "stop";
}

export function hasScoringError(tr: MockTaskResult, i: number): boolean {
  if (tr.task.type !== "code" && tr.task.type !== "judge" && tr.task.type !== "omni") return false;
  return ((instanceData(tr).u[i] * 7919) % 1) < 0.012;
}

// ---------------------------------------------------------------------------------------- scores

export interface TaskScore {
  score: number | null;
  stderr: number | null;
  n: number;
  metrics: Record<string, number | null>;
}

const scoreCache = new Map<number, TaskScore>();

export function taskScore(tr: MockTaskResult): TaskScore {
  const cached = scoreCache.get(tr.id);
  if (cached) return cached;
  if (tr.error) {
    const empty: TaskScore = { score: null, stderr: null, n: 0, metrics: {} };
    scoreCache.set(tr.id, empty);
    return empty;
  }
  const data = instanceData(tr);
  const n = tr.n;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sum += data.primary[i];
    sumSq += data.primary[i] * data.primary[i];
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const metrics: Record<string, number | null> = {};
  for (const key of tr.task.metrics) {
    if (key === tr.task.primary) {
      metrics[key] = mean * tr.task.scale;
      continue;
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += instanceMetrics(tr, i)[key] ?? 0;
    metrics[key] = s / n;
  }
  const result: TaskScore = {
    score: mean * tr.task.scale,
    stderr: (Math.sqrt(variance) / Math.sqrt(n)) * tr.task.scale,
    n,
    metrics,
  };
  scoreCache.set(tr.id, result);
  return result;
}

export interface SuiteScore {
  score: number | null;
  stderr: number | null;
  numTasks: number;
  nInstances: number;
  missing: string[];
}

/** Aggregate a suite over a task-name → task result lookup (average or average of averages). */
export function suiteScore(name: string, lookup: (task: string) => MockTaskResult | undefined): SuiteScore {
  const def = SUITE_BY_NAME.get(name);
  if (!def) return { score: null, stderr: null, numTasks: 0, nInstances: 0, missing: [] };
  const parts: { score: number; stderr: number | null }[] = [];
  let numTasks = 0;
  let nInstances = 0;
  const missing: string[] = [];
  for (const child of def.children) {
    if (child.type === "task") {
      const tr = lookup(child.name);
      const s = tr ? taskScore(tr) : null;
      if (s && s.score != null) {
        parts.push({ score: s.score, stderr: s.stderr });
        numTasks += 1;
        nInstances += s.n;
      } else missing.push(child.name);
    } else {
      const sub = suiteScore(child.name, lookup);
      missing.push(...sub.missing);
      if (sub.score != null) {
        parts.push({ score: sub.score, stderr: sub.stderr });
        numTasks += sub.numTasks;
        nInstances += sub.nInstances;
      }
    }
  }
  if (!parts.length) return { score: null, stderr: null, numTasks, nInstances, missing };
  const w = 1 / parts.length;
  const score = parts.reduce((acc, p) => acc + p.score * w, 0);
  const stderr = Math.sqrt(parts.reduce((acc, p) => acc + (p.stderr ?? 0) ** 2 * w * w, 0));
  return { score, stderr, numTasks, nInstances, missing };
}

export function runLookup(run: MockRun): (task: string) => MockTaskResult | undefined {
  const w = getWorld();
  const map = new Map<string, MockTaskResult>();
  for (const id of run.taskResultIds) {
    const tr = w.taskResults.get(id)!;
    map.set(tr.task.name, tr);
  }
  return (task) => map.get(task);
}

function computeHeadline(run: MockRun): Headline | null {
  if (!run.taskResultIds.length) return null;
  const lookup = runLookup(run);
  const tasks = new Set(run.taskResultIds.map((id) => getWorld().taskResults.get(id)!.task.name));
  for (const top of ["olmes:base", "math:olmo3", "safety:olmo3"]) {
    if (suiteLeaves(top).some((t) => tasks.has(t))) {
      const s = suiteScore(top, lookup);
      if (s.score != null) {
        return { kind: "suite", name: top, score: s.score, stderr: s.stderr, display_format: "percent" };
      }
    }
  }
  const scores = run.taskResultIds
    .map((id) => getWorld().taskResults.get(id)!)
    .filter((tr) => tr.task.kind !== "unbounded")
    .map((tr) => taskScore(tr).score)
    .filter((v): v is number => v != null);
  if (!scores.length) return null;
  return {
    kind: "mean",
    name: null,
    score: scores.reduce((a, b) => a + b, 0) / scores.length,
    stderr: null,
    display_format: "percent",
  };
}
