"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from typing import Optional

from catwalk.dependencies.lm_eval.base import MultipleChoiceTask

from .std_fewshot import STD_FEWSHOT
from .utils import make_cloze_prompt, make_mcq_prompt

_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""


class ARCEasyStd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Easy"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs: Optional[list] = list(
                map(self._process_doc, self.dataset["train"])
            )
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        # Question: When a switch is used in an electrical circuit, the switch can
        # Answer: stop and start the flow of current.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        query = make_cloze_prompt(doc["question"])
        out_doc = {
            "id": doc["id"],
            "query": query,
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

    def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs: Optional[list] = list(
                map(self._process_doc, STD_FEWSHOT[self.DATASET_NAME])
            )
        assert k <= len(self._fewshot_docs)
        return self._fewshot_docs[:k]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class ARCEasyMCStd(ARCEasyStd):
    # Include answer choices in prompt, answer is just the single letter A, B, ... E.g.,
    # Question: When a switch is used in an electrical circuit, the switch can
    #  A. cause the charge to build.
    #  B. increase and decrease the voltage.
    #  C. cause the current to change direction.
    #  D. stop and start the flow of current.
    # Answer: D

    def _process_doc(self, doc):
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        num_choices = len(doc["choices"]["text"])
        choice_labels = ["A", "B", "C", "D", "E"][:num_choices]
        query = make_mcq_prompt(doc["question"], doc["choices"]["text"])
        out_doc = {
            "id": doc["id"],
            "query": query,
            "choices": choice_labels,
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

    def unconditioned_prompt(self):
        # Don't need unconditioned normalization here
        return None


class ARCChallengeStd(ARCEasyStd):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"


class ARCChallengeMCStd(ARCEasyMCStd):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
