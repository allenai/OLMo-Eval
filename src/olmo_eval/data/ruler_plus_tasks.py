"""RULER-plus (ruler-plus) task configurations.

Mirrors ruler_tasks.py, but describes the extended-context RULER dataset
generated with https://github.com/jopetty/RULER (scripts/generate-data.sh)
and downloaded from the allenai/ruler-plus HuggingFace dataset repo (see
ruler_plus_loader.py) instead of the allenai/ruler_data HuggingFace release.
"""

# Context sizes to generate tasks for, including sizes beyond the original 131072 cap
CONTEXT_SIZES = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152)

# Default configuration values
_DEFAULT_MAX_GEN_TOKS = 50

# Base task configurations. data_template is relative to the dataset root
# (see ruler_plus_loader.get_ruler_plus_data_file) and takes a {size} placeholder.
# max_gen_toks follows upstream RULER's per-task tokens_to_generate, not the
# HELMET budgets that ruler_tasks.py uses.
_BASE_TASKS = {
    # NIAH (Needle in a Haystack) - Single variants
    "niah_s_1": {
        "data_template": "{size}/niah_single_1/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    "niah_s_2": {
        "data_template": "{size}/niah_single_2/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    "niah_s_3": {
        "data_template": "{size}/niah_single_3/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    # NIAH - Multi-key variants
    "niah_mk_1": {
        "data_template": "{size}/niah_multikey_1/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    "niah_mk_2": {
        "data_template": "{size}/niah_multikey_2/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    "niah_mk_3": {
        "data_template": "{size}/niah_multikey_3/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    # NIAH - Multi-value variant
    "niah_mv": {
        "data_template": "{size}/niah_multivalue/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    # NIAH - Multi-query variant
    "niah_mq": {
        "data_template": "{size}/niah_multiquery/validation.jsonl",
        "max_gen_toks": 128,
        "tag": "niah",
    },
    # Multi-hop tracing - Variable tracking
    "vt": {
        "data_template": "{size}/vt/validation.jsonl",
        "max_gen_toks": 30,
        "tag": "multi_hop_tracing",
    },
    # Aggregation - Common word extraction
    "cwe": {
        "data_template": "{size}/cwe/validation.jsonl",
        "max_gen_toks": 120,
        "tag": "aggregation",
    },
    # Aggregation - Frequency word extraction
    "fwe": {
        "data_template": "{size}/fwe/validation.jsonl",
        "max_gen_toks": 50,
        "tag": "aggregation",
    },
    # Question Answering
    "qa_1": {
        "data_template": "{size}/qa_1/validation.jsonl",
        "max_gen_toks": 32,
        "tag": "qa",
    },
    "qa_2": {
        "data_template": "{size}/qa_2/validation.jsonl",
        "max_gen_toks": 32,
        "tag": "qa",
    },
}


def _generate_ruler_plus_tasks() -> dict:
    """Generate RULER_PLUS_TASKS dictionary from base configurations.

    Creates task definitions for all combinations of base tasks and context sizes.

    Returns:
        Dictionary mapping task names to their configurations.
    """
    tasks = {}

    for task_type, base_config in _BASE_TASKS.items():
        for size in CONTEXT_SIZES:
            task_name = f"{task_type}__{size}"

            tasks[task_name] = {
                "data": str(base_config["data_template"]).format(size=size),
                "max_gen_toks": base_config.get("max_gen_toks", _DEFAULT_MAX_GEN_TOKS),
                "use_chat_template": False,
                "stop_new_line": False,
                "tag": base_config["tag"],
            }

    return tasks


# Generate the full RULER_PLUS_TASKS dictionary
RULER_PLUS_TASKS = _generate_ruler_plus_tasks()
