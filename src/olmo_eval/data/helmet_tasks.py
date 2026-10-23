"""HELMET task configurations, covering all seven of HELMET's categories.

Recall (json_kv), RAG (KILT), re-ranking (MS MARCO), LongQA (InfiniteBench,
NarrativeQA), summarization (InfiniteBench, Multi-LexSum), ICL, and citation
(ALCE), at HELMET's standard 4k-128k context lengths.

Each task declares its own `context_sizes` so that tasks whose length is
bounded by real documents, fixed retrieval depth, or finite demonstration
pools can stop short of tasks whose context is synthetic.

Tasks are generated programmatically from base configurations, following the
same pattern as `ruler_tasks.py`.
"""

# Length tiers published in helmet-plus, mapping the manifest key to its
# resolved token length (see HELMET's scripts/generate_json_kv_data.py).
LENGTH_NAMES = {
    4096: "4k",
    8192: "8k",
    16384: "16k",
    32768: "32k",
    65536: "64k",
    131072: "128k",
}

# Standard HELMET lengths.
STANDARD_CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072]
CONTEXT_SIZES = list(STANDARD_CONTEXT_SIZES)

_DEFAULT_MAX_GEN_TOKS = 100
_DEFAULT_SHOTS = 2
_DEFAULT_LIMIT = 100

# Shot counts per ICL dataset, aligned with STANDARD_CONTEXT_SIZES. These are
# HELMET's own values (scripts/generate_configs.py): for ICL the number of
# demonstrations, not truncation, is what sets the context length.
#
# Caveat worth knowing when reading results: HELMET fit these against a
# Llama-2-era tokenizer, so under Olmo 3's larger vocabulary the rendered
# prompts come out at ~0.78-0.85x their nominal length -- "icl_*__4096" is
# really ~3.3-3.5k tokens, and "__131072" is closer to 105k. They are kept
# as-is anyway so our ICL numbers stay directly comparable to published
# HELMET results; the alternative trades that comparability for internal
# consistency with json_kv, which is calibrated against Olmo 3 and does hit
# its nominal lengths. Run scripts/internal/calibrate_helmet_icl_shots.py to
# measure the current shortfall or to generate Olmo-3-calibrated counts.
#
# Note also that trec has only 5452 training examples, so its longest tiers
# repeat demonstrations (balance_labels samples with replacement).
_ICL_SHOTS = {
    "trec_coarse": [200, 400, 800, 1600, 3300, 6600],
    "trec_fine": [200, 400, 800, 1600, 3200, 6400],
    "banking77": [180, 360, 720, 1450, 2900, 5900],
    "clinic150": [220, 440, 880, 1750, 3525, 7050],
    "nlu": [250, 510, 1020, 2040, 4080, 8296],
}

# HELMET evaluates ICL on 500 samples (configs/icl.yaml) rather than the 100
# used for the recall tasks.
_ICL_LIMIT = 500
_ICL_MAX_GEN_TOKS = 20

# LongQA reserves part of each length tier for the prompt and the generation,
# and gives the rest to the story: HELMET names these tasks with a postfix of
# `length - 200 - generation_max_length` and truncates the context to exactly
# that many reference-tokenizer tokens (see generate_configs.py).
_LONGQA_PROMPT_RESERVE = 200
_INFBENCH_QA_MAX_GEN_TOKS = 10
_NARRATIVEQA_MAX_GEN_TOKS = 100

# RAG answers are short spans (configs/rag.yaml)
_RAG_MAX_GEN_TOKS = 20

# re-ranking emits a full `ID3 > ID1 > ...` ordering, so it needs far more room
_RERANK_MAX_GEN_TOKS = 200

# summarization writes paragraphs, and multi_lexsum reserves 300 tokens of the
# tier for prompt and buffer where the other long-context tasks reserve 200
_INFBENCH_SUM_MAX_GEN_TOKS = 1200
_MULTI_LEXSUM_MAX_GEN_TOKS = 400
_MULTI_LEXSUM_PROMPT_RESERVE = 300

# ALCE writes a cited paragraph; the nocite variants are given more room
# because they are run zero-shot (configs/alce_nocite.yaml)
_ALCE_MAX_GEN_TOKS = 300
_ALCE_NOCITE_MAX_GEN_TOKS = 600

# HELMET's recall category pairs json_kv with three RULER needle-in-a-haystack
# tasks (configs/recall.yaml), with these generation budgets. RULER's own
# budget for niah_mv is longer; this keeps HELMET's.
_RECALL_RULER_MAX_GEN_TOKS = {"niah_mk_2": 50, "niah_mk_3": 100, "niah_mv": 50}

# Base task configurations, keyed by HELMET task type.
_BASE_TASKS: dict[str, dict] = {
    "json_kv": {
        "kind": "json_kv",
        "tag": "recall",
        "context_sizes": CONTEXT_SIZES,
    },
    **{
        f"ruler_{name}": {
            "kind": "ruler",
            "tag": "recall",
            "ruler_task": name,
            "context_sizes": STANDARD_CONTEXT_SIZES,
            "max_gen_toks": gen_toks,
            # RULER prompts carry no demonstrations
            "shots": 0,
        }
        for name, gen_toks in _RECALL_RULER_MAX_GEN_TOKS.items()
    },
    **{
        f"icl_{name}": {
            "kind": "icl",
            "tag": "icl",
            "icl_dataset": name,
            "context_sizes": STANDARD_CONTEXT_SIZES,
            "shots_by_size": dict(zip(STANDARD_CONTEXT_SIZES, shots, strict=True)),
            "max_gen_toks": _ICL_MAX_GEN_TOKS,
            "limit": _ICL_LIMIT,
            # HELMET sets stop_new_line=True for ICL: the answer is a single
            # "label: N" line, so generation should stop at the newline.
            "stop_new_line": True,
        }
        for name, shots in _ICL_SHOTS.items()
    },
    "infbench_qa_eng": {
        "kind": "infbench",
        "tag": "longqa",
        "infbench_subset": "qa_eng",
        # capped at standard HELMET's lengths: the context is a real book, so
        # it can't be stretched the way json_kv's synthetic context can
        "context_sizes": STANDARD_CONTEXT_SIZES,
        "max_gen_toks": _INFBENCH_QA_MAX_GEN_TOKS,
        "shots": 2,
        # HELMET runs LongQA with a chat template, so the "Answer:" prefix is
        # not appended to the prompt as a partial assistant turn
        "use_chat_template": True,
    },
    **{
        f"kilt_{name}": {
            "kind": "kilt",
            "tag": "rag",
            "kilt_task": f"kilt_{name}",
            # retrieval depth is baked into the pre-retrieved files, so these
            # cannot be pushed past standard HELMET's lengths
            "context_sizes": STANDARD_CONTEXT_SIZES,
            "max_gen_toks": _RAG_MAX_GEN_TOKS,
            "shots": 2,
            # the answer is a single short line, so stop at the newline
            "stop_new_line": True,
            # Each question repeats once per gold-passage depth, and the score
            # averages over depths. The loader caps questions; an instance
            # `limit` on top would re-sample rows and break up the sweep.
            "max_questions": _DEFAULT_LIMIT,
            "limit": None,
            **extra,
        }
        for name, extra in {
            "nq": {},
            "triviaqa": {},
            "hotpotqa": {},
            # HELMET evaluates PopQA restricted to long-tail entities, encoding
            # the cutoff in the task name (kilt_popqa_3 -> log10(s_pop) < 3)
            "popqa": {"popularity_threshold": 3.0},
        }.items()
    },
    "msmarco_rerank_psg": {
        "kind": "msmarco",
        "tag": "rerank",
        # candidate-list depth is fixed in the pre-retrieved files
        "context_sizes": STANDARD_CONTEXT_SIZES,
        "max_gen_toks": _RERANK_MAX_GEN_TOKS,
        "shots": 2,
        "stop_new_line": True,
    },
    "infbench_sum_eng": {
        "kind": "infbench",
        "metrics_key": "summ_book",
        "tag": "summ",
        "infbench_subset": "sum_eng",
        "context_sizes": STANDARD_CONTEXT_SIZES,
        "max_gen_toks": _INFBENCH_SUM_MAX_GEN_TOKS,
        "shots": 2,
        "use_chat_template": True,
        "judged": True,
    },
    "multi_lexsum": {
        "kind": "multi_lexsum",
        "metrics_key": "summ_lawsuit",
        "tag": "summ",
        "context_sizes": STANDARD_CONTEXT_SIZES,
        "max_gen_toks": _MULTI_LEXSUM_MAX_GEN_TOKS,
        "prompt_reserve": _MULTI_LEXSUM_PROMPT_RESERVE,
        "shots": 2,
        "use_chat_template": True,
        "judged": True,
    },
    **{
        name: {
            "kind": "alce",
            "metrics_key": metrics_key,
            "tag": "cite",
            "alce_task": name,
            # HELMET's Cite average leaves out the nocite ablations, so they are
            # registered but kept out of the category suites
            "ablation": name.endswith("_nocite"),
            # every tier reads the same 2000-document pool and shows a prefix
            # of it, so the ceiling is where the pool runs out
            "context_sizes": STANDARD_CONTEXT_SIZES,
            "max_gen_toks": gen_toks,
            "shots": shots,
            "use_chat_template": True,
        }
        for name, metrics_key, gen_toks, shots in [
            ("alce_asqa", "alce_asqa", _ALCE_MAX_GEN_TOKS, 2),
            ("alce_qampari", "alce_qampari", _ALCE_MAX_GEN_TOKS, 2),
            ("alce_asqa_nocite", "alce_asqa", _ALCE_NOCITE_MAX_GEN_TOKS, 0),
            ("alce_qampari_nocite", "alce_qampari", _ALCE_NOCITE_MAX_GEN_TOKS, 0),
        ]
    },
    "narrativeqa": {
        "kind": "narrativeqa",
        # graded by an LLM judge rather than string overlap, so it only runs
        # where a judge is configured -- see suites/helmet.py
        "metrics_key": "narrativeqa",
        "tag": "longqa",
        "context_sizes": STANDARD_CONTEXT_SIZES,
        "max_gen_toks": _NARRATIVEQA_MAX_GEN_TOKS,
        "shots": 2,
        "use_chat_template": True,
        "judged": True,
    },
    "infbench_choice_eng": {
        "kind": "infbench",
        # same loader and task class as qa_eng, but scored by exact match on
        # the chosen letter rather than ROUGE
        "metrics_key": "infbench_choice",
        "tag": "longqa",
        "infbench_subset": "choice_eng",
        "context_sizes": STANDARD_CONTEXT_SIZES,
        "max_gen_toks": _INFBENCH_QA_MAX_GEN_TOKS,
        "shots": 2,
        "use_chat_template": True,
    },
}


def _generate_helmet_tasks() -> dict:
    """Generate HELMET_TASKS dictionary from base configurations.

    Creates task definitions for each base task at each of the context sizes
    that task actually supports.

    Returns:
        Dictionary mapping task names to their configurations.
    """
    tasks = {}

    for task_type, base_config in _BASE_TASKS.items():
        for size in base_config.get("context_sizes", CONTEXT_SIZES):
            task_name = f"{task_type}__{size}"

            shots_by_size = base_config.get("shots_by_size")
            shots = (
                shots_by_size[size]
                if shots_by_size is not None
                else base_config.get("shots", _DEFAULT_SHOTS)
            )

            max_gen_toks = base_config.get("max_gen_toks", _DEFAULT_MAX_GEN_TOKS)

            task: dict = {
                "kind": base_config["kind"],
                # which metric set to score with; defaults to the task kind,
                # but subsets sharing a loader can score differently
                "metrics_key": base_config.get("metrics_key", base_config["kind"]),
                "length_name": LENGTH_NAMES[size],
                "shots": shots,
                "max_gen_toks": max_gen_toks,
                "limit": base_config.get("limit", _DEFAULT_LIMIT),
                "tag": base_config["tag"],
            }
            if base_config.get("judged"):
                task["judged"] = True
            if base_config["kind"] in ("narrativeqa", "multi_lexsum"):
                reserve = base_config.get("prompt_reserve", _LONGQA_PROMPT_RESERVE)
                task["max_context_tokens"] = size - reserve - max_gen_toks
            if "alce_task" in base_config:
                task["alce_task"] = base_config["alce_task"]
            if base_config.get("ablation"):
                task["ablation"] = True
            if "ruler_task" in base_config:
                task["ruler_task"] = base_config["ruler_task"]
            if "kilt_task" in base_config:
                task["kilt_task"] = base_config["kilt_task"]
                # HELMET trims the retrieved passages at inference time so the
                # prompt plus generation fits the tier
                task["max_prompt_tokens"] = size - max_gen_toks
                task["max_questions"] = base_config["max_questions"]
                if "popularity_threshold" in base_config:
                    task["popularity_threshold"] = base_config["popularity_threshold"]
            if "icl_dataset" in base_config:
                task["icl_dataset"] = base_config["icl_dataset"]
            if "infbench_subset" in base_config:
                task["infbench_subset"] = base_config["infbench_subset"]
                # the story gets whatever the prompt and the generation don't
                task["max_context_tokens"] = size - _LONGQA_PROMPT_RESERVE - max_gen_toks
            if base_config.get("stop_new_line"):
                task["stop_new_line"] = True
            if base_config.get("use_chat_template"):
                task["use_chat_template"] = True

            tasks[task_name] = task

    return tasks


# Generate the full HELMET_TASKS dictionary
HELMET_TASKS = _generate_helmet_tasks()
