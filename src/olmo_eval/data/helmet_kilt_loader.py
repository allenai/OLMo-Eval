"""HELMET RAG (KILT) data loading.

Ports HELMET's `load_qa` (https://github.com/princeton-nlp/HELMET, data.py):
an open-domain question plus a stack of retrieved Wikipedia passages, where the
model must answer from the passages.

Context length is set primarily by how many passages were retrieved, so each
length tier is a separate pre-retrieved file. The retrieved passages
overshoot the tier's budget, though, and HELMET relies on trimming the end of
the passage block at inference time (`tokenize` in its model_utils.py); that
trim is reproduced here with the same fixed reference tokenizer the other
long-context loaders use, so the prompt a model sees is independent of its own
tokenizer. That retrieval was run
once by the HELMET authors and is re-hosted unpacked on `allenai/helmet-plus`;
see `kilt/manifest.json` there for the tier -> file mapping. These files are
large (the 128k tiers are 1-3 GB each) but are cached by huggingface_hub after
the first download.

Because retrieval depth is fixed in the data, these tasks cannot be extended
past standard HELMET's 128k the way the synthetic recall task can.

**`max_samples` caps questions, not instances.** Each question appears several
times in these files with the gold passage planted at different relative
depths -- 6 for nq/triviaqa/popqa (the `dep6` in the filenames: 0.0, 0.2, 0.4,
0.6, 0.8, 0.95) and 3 for hotpotqa -- so the score averages over where in the
context the answer sits, which is the "lost in the middle" effect these tasks
exist to measure. HELMET samples by question and keeps every depth for the
questions it picks, and that is reproduced here: 100 questions yield ~600
instances for nq, not 100. Budget accordingly at the longer tiers. The task
passes its `max_questions` setting here and leaves the runner's instance
`limit` unset, so the sweep reaches the model intact.
"""

import hashlib
import json
import logging
import random
from typing import Any

from olmo_eval.data.helmet_infbench_loader import REFERENCE_TOKENIZER, _load_reference_tokenizer
from olmo_eval.data.helmet_loader import download_helmet_plus_file, sample_jsonl_by_key

logger = logging.getLogger(__name__)

KILT_MANIFEST = "kilt/manifest.json"

_USER_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}{context}\n\nQuestion: {question}"
)
_SYSTEM_TEMPLATE = "Answer:"
_PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
_DEMO_TEMPLATE = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"


def load_kilt_manifest() -> dict[str, dict[str, Any]]:
    """Download and parse the KILT manifest: task -> length tier -> file paths."""
    with open(download_helmet_plus_file(KILT_MANIFEST), encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _question_key(row: dict[str, Any]) -> Any:
    return row["id"] if "id" in row else row["question"]


def _render_passages(contexts: list[dict[str, Any]]) -> str:
    return "\n\n".join(_PASSAGE_TEMPLATE.format(**ctx) for ctx in contexts)


def _instance_seed(key_value: Any) -> int:
    """Per-instance seed for demo selection.

    hashlib rather than hash() because the latter is salted per process; this
    matches HELMET, so an instance draws the same demos on every run.
    """
    return int(hashlib.sha256(str(key_value).encode("utf-8")).hexdigest(), 16) % 2**31


def _drop_duplicates(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen, out = set(), []
    for record in records:
        if record[key] in seen:
            continue
        seen.add(record[key])
        out.append(record)
    return out


def _truncate_prompt_context(
    sample: dict[str, Any], prompt_template: str, max_prompt_tokens: int, tokenizer
) -> dict[str, Any]:
    """Trim the end of `context` until the rendered prompt fits `max_prompt_tokens`.

    Mirrors HELMET's `tokenize`: the overflow is measured on the full prompt and
    removed from the passage block alone, so the instruction, demonstrations and
    question survive intact.
    """
    prompt_len = len(tokenizer(prompt_template.format(**sample))["input_ids"])
    overflow = prompt_len - max_prompt_tokens
    if overflow <= 0 or not sample["context"]:
        return sample
    offsets = tokenizer(sample["context"], return_offsets_mapping=True)["offset_mapping"]
    cut_at = offsets[-overflow][0] if overflow < len(offsets) else 0
    return {**sample, "context": sample["context"][:cut_at]}


def load_kilt_dataset(
    task: str,
    length_name: str,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
    popularity_threshold: float | None = None,
    max_prompt_tokens: int | None = None,
    reference_tokenizer: str = REFERENCE_TOKENIZER,
) -> dict[str, Any]:
    """Load a HELMET RAG dataset for one task at one length tier.

    Args:
        task: Manifest key, e.g. "kilt_nq" or "kilt_popqa".
        length_name: Tier key, e.g. "4k" or "128k".
        shots: Number of worked examples to prepend. Demos carry their own
            (much shorter) passage sets, drawn from the k3 demo file.
        max_samples: Cap on the number of instances.
        seed: Seed for instance sampling. Demo selection is additionally keyed
            per instance, so it is stable regardless of this cap.
        popularity_threshold: PopQA only -- keep entities whose subject
            popularity is below 10^threshold, which is how HELMET restricts the
            task to genuinely long-tail entities.
        max_prompt_tokens: Budget for the rendered prompt, measured with the
            reference tokenizer. Passages are trimmed from the end until the
            prompt fits; None disables the trim.
        reference_tokenizer: Tokenizer used for the budget. Changing it changes
            how much text every model sees.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    manifest = load_kilt_manifest()
    if task not in manifest:
        raise ValueError(f"Unknown KILT task '{task}'. Available: {sorted(manifest)}")
    if length_name not in manifest[task]:
        raise ValueError(
            f"Unknown length '{length_name}' for {task}. Available: {sorted(manifest[task])}"
        )

    entry = manifest[task][length_name]
    logger.info("Fetching %s data for %s (%s)...", task, length_name, entry["test_file"])
    test_path = download_helmet_plus_file(entry["test_file"])
    demo_pool = _load_jsonl(download_helmet_plus_file(entry["demo_file"]))

    below_threshold = None
    if popularity_threshold is not None:
        import math

        def below_threshold(record: dict[str, Any]) -> bool:
            return math.log10(record["s_pop"]) < popularity_threshold

        demo_pool = [r for r in demo_pool if below_threshold(r)]

    data = sample_jsonl_by_key(
        test_path, max_samples, seed, key=_question_key, keep=below_threshold
    )
    if not data:
        raise ValueError(f"No rows loaded for {task} at {length_name} from {test_path}")

    # some sources have no id, in which case the question stands in as the key
    key = "id" if "id" in data[0] else "question"

    def build(sample: dict[str, Any]) -> dict[str, Any]:
        demo_text = ""
        if shots > 0:
            candidates = demo_pool
            if popularity_threshold is not None:
                # PopQA's demo pool is the test file itself, so the instance
                # being answered has to be excluded from its own demos
                candidates = [d for d in demo_pool if d[key] != sample[key]]

            shuffled = list(candidates)
            random.Random(_instance_seed(sample[key])).shuffle(shuffled)
            demos = _drop_duplicates(shuffled, key)[:shots]

            demo_text = (
                "\n\n".join(
                    _DEMO_TEMPLATE.format(
                        documents=_render_passages(d["ctxs"]),
                        question=d["question"],
                        answer=d["answers"][0],
                    )
                    for d in demos
                )
                + "\n\n"
            )

        return {
            "demos": demo_text,
            "context": _render_passages(sample["ctxs"]) if sample.get("ctxs") else "",
            "question": sample["question"],
            "answer": sample["answers"],
        }

    prompt_template = _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE
    rows = [build(r) for r in data]
    if max_prompt_tokens is not None:
        tokenizer = _load_reference_tokenizer(reference_tokenizer)
        rows = [
            _truncate_prompt_context(row, prompt_template, max_prompt_tokens, tokenizer)
            for row in rows
        ]

    return {
        "data": rows,
        "prompt_template": prompt_template,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
