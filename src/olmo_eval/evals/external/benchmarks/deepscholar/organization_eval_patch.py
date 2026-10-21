"""Replacement for DeepScholar's organization evaluator.

This file is copied into the pinned DeepScholar checkout at runtime. It targets
the pinned LOTUS revision, whose ``pairwise_judge`` accessor returns normalized
``"A"``/``"B"`` strings rather than the Pydantic response objects expected by
the upstream evaluator.
"""

from __future__ import annotations

import lotus  # type: ignore[ty:unresolved-import]  # noqa: F401
import pandas as pd

try:
    from evaluator import EvaluationFunction, Evaluator  # type: ignore[ty:unresolved-import]
    from parsers import Parser  # type: ignore[ty:unresolved-import]
except ImportError:
    from ..parsers import Parser  # type: ignore[ty:unresolved-import]
    from .enum import EvaluationFunction  # type: ignore[ty:unresolved-import]
    from .evaluator import Evaluator  # type: ignore[ty:unresolved-import]


ORGANIZATION_JUDGE_INSTRUCTION = """
You will receive the title and abstract of a research paper together with two
candidate related-work sections written for that paper.

Decide which section exhibits better organization and coherence. Ignore the
text's Markdown or LaTeX formatting and assess organization only:

- Logical structure: a clear introduction, sensible thematic grouping, and a
  smooth progression of ideas.
- Paragraph cohesion: each paragraph develops one topic and flows naturally to
  the next.
- Clarity and readability: minimal redundancy or contradiction, with useful
  transitions.
- Signposting: helpful headings, topic sentences, or discourse markers.

Do not judge breadth, citation accuracy, or analytic depth. There are no ties.
Return exactly the choice token requested by the surrounding instruction. Do
not return JSON or an explanation.

### Paper under assessment
{paper_title}
{paper_abstract}
"""


def _strict_decision(raw_output: object) -> str | None:
    """Parse only a bare A/B judge response, never LOTUS's fallback decision."""
    if not isinstance(raw_output, str):
        return None
    decision = raw_output.strip().upper()
    return decision if decision in {"A", "B"} else None


def _raw_output_column(results: pd.DataFrame, suffix: str) -> pd.Series:
    """Return the raw-output column emitted by a one-trial LOTUS judge call."""
    column = f"raw_output{suffix}_0"
    if column not in results:
        raise RuntimeError(
            f"organization judge did not return expected raw output column {column!r}; "
            f"got {list(results.columns)!r}"
        )
    return results[column]


class OrganizationEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.ORGANIZATION

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        system_prompt = (
            "You are an intelligent, rigorous, and fair evaluator of scholarly "
            "writing. Follow the requested output format exactly and return only "
            "the single choice token."
        )
        infos = [parser.get_folder_info(include_related_works_section=True) for parser in parsers]
        df = pd.DataFrame(infos)
        df.rename(
            columns={
                "generated_related_works_section": "related_work_a",
                "related_works_section": "related_work_b",
            },
            inplace=True,
        )

        # Run the two orders separately. Pinned LOTUS converts malformed outputs
        # to a boolean default before exposing its A/B decision column, which can
        # silently turn parse failures into generated-section wins. Its raw-output
        # columns preserve the judge text, so parse those strictly instead.
        forward: pd.DataFrame = df.pairwise_judge(
            col1="related_work_a",
            col2="related_work_b",
            judge_instruction=ORGANIZATION_JUDGE_INSTRUCTION,
            system_prompt=system_prompt,
            n_trials=1,
            permute_cols=False,
            return_raw_outputs=True,
            suffix="_judge_forward",
        )
        reverse: pd.DataFrame = df.pairwise_judge(
            col1="related_work_b",
            col2="related_work_a",
            judge_instruction=ORGANIZATION_JUDGE_INSTRUCTION,
            system_prompt=system_prompt,
            n_trials=1,
            permute_cols=False,
            return_raw_outputs=True,
            suffix="_judge_reverse",
        )

        raw_forward = _raw_output_column(forward, "_judge_forward")
        raw_reverse = _raw_output_column(reverse, "_judge_reverse")
        decision_forward = raw_forward.map(_strict_decision)
        decision_reverse = raw_reverse.map(_strict_decision)
        invalid = decision_forward.isna() | decision_reverse.isna()
        if invalid.any():
            examples = [
                {
                    "folder_path": df.loc[index, "folder_path"],
                    "forward": raw_forward.loc[index],
                    "reverse": raw_reverse.loc[index],
                }
                for index in df.index[invalid][:5]
            ]
            raise ValueError(
                "organization judge returned non-A/B output for "
                f"{int(invalid.sum())}/{len(df)} rows; examples={examples!r}"
            )

        results = df.copy()
        results["organization_raw_v1"] = raw_forward
        results["organization_raw_v2"] = raw_reverse
        # Forward A selects generated. Reverse B selects generated.
        results["organization_v1"] = (decision_forward == "A").astype(int)
        results["organization_v2"] = (decision_reverse == "B").astype(int)
        results[self.evaluation_function.value] = (
            results["organization_v1"] + results["organization_v2"]
        ) / 2
        return results
