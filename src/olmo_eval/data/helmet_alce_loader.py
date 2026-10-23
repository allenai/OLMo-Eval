"""HELMET Cite (ALCE) data loading.

Ports HELMET's `load_alce` (https://github.com/princeton-nlp/HELMET, data.py):
a question plus a stack of retrieved documents, where the model must answer
*and* cite the documents it used.

Context length is set by how many of the retrieved documents are shown. Every
tier reads the same file -- a fixed pool of 2000 GTR-retrieved documents per
question -- and takes a prefix of it, so unlike the RAG tasks there is one data
file rather than one per tier. The ceiling is still standard HELMET's 128k,
since the pool runs out.

Data and the few-shot prompt files come from `allenai/helmet-plus`; see
`alce/manifest.json` there for the tier -> document-count mapping.
"""

import json
import logging
import random
from typing import Any

from olmo_eval.data.helmet_loader import download_helmet_plus_file

logger = logging.getLogger(__name__)

ALCE_MANIFEST = "alce/manifest.json"

_USER_TEMPLATE = "{demo_text}{instruction}\n\nQuestion: {question}\n\n{context}"
_SYSTEM_TEMPLATE = "Answer:"


def load_alce_manifest() -> dict[str, dict[str, Any]]:
    with open(download_helmet_plus_file(ALCE_MANIFEST), encoding="utf-8") as f:
        return json.load(f)


def load_alce_dataset(
    task: str,
    length_name: str,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load an ALCE task at one length tier.

    Args:
        task: Manifest key, e.g. "alce_asqa" or "alce_qampari_nocite".
        length_name: Tier key, e.g. "4k" or "128k".
        shots: Number of worked examples to prepend. These come from the prompt
            file rather than the dataset, and carry their own documents.
        max_samples: Cap on the number of instances.
        seed: Seed for instance sampling and demo selection.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    manifest = load_alce_manifest()
    if task not in manifest:
        raise ValueError(f"Unknown ALCE task '{task}'. Available: {sorted(manifest)}")
    if length_name not in manifest[task]:
        raise ValueError(
            f"Unknown length '{length_name}' for {task}. Available: {sorted(manifest[task])}"
        )

    entry = manifest[task][length_name]
    num_docs = entry["num_passages"]

    with open(download_helmet_plus_file(entry["test_file"]), encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate either shape; asqa/qampari ship as arrays
        data = data["data"]

    with open(download_helmet_plus_file(entry["demo_file"]), encoding="utf-8") as f:
        prompt_spec = json.load(f)

    instruction = prompt_spec["instruction"]
    demo_prompt = prompt_spec["demo_prompt"]
    doc_prompt = prompt_spec["doc_prompt"]

    def render_docs(docs: list[dict[str, Any]], limit: int | None = None) -> str:
        chosen = docs[:limit] if limit is not None else docs
        return "\n\n".join(doc_prompt.format(**doc, ID=idx + 1) for idx, doc in enumerate(chosen))

    demo_text = ""
    if shots > 0:
        # upstream samples the worked examples with the global RNG, so its
        # prompt varies run to run; seeding here makes it reproducible
        chosen = random.Random(seed).sample(prompt_spec["demos"], shots)
        demo_text = (
            "\n\n\n".join(
                demo_prompt.format(
                    **demo, instruction=instruction, context=render_docs(demo["docs"])
                )
                for demo in chosen
            )
            + "\n\n\n"
        )

    if max_samples is not None and len(data) > max_samples:
        shuffled = list(data)
        random.Random(seed).shuffle(shuffled)
        data = shuffled[:max_samples]

    records = []
    for item in data:
        record = {
            "context": render_docs(item["docs"], num_docs),
            "demo_text": demo_text,
            "instruction": instruction,
            "question": item["question"],
            "answer": item.get("answer"),
        }
        # scoring inputs, kept separate from the prompt: ASQA scores coverage of
        # its disambiguated sub-questions, QAMPARI recall over an answer set
        if item.get("qa_pairs"):
            record["qa_pairs"] = item["qa_pairs"]
        if item.get("answers"):
            record["qampari_answers"] = item["answers"]
        records.append(record)

    return {
        "data": records,
        "prompt_template": _USER_TEMPLATE + "\n\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
