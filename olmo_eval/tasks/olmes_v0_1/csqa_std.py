"""
Commonsense QA
"""

from typing import Optional

from catwalk.dependencies.lm_eval.base import MultipleChoiceTask

from .std_fewshot import STD_FEWSHOT
from .utils import make_cloze_prompt, make_mcq_prompt

_CITATION = """
@inproceedings{talmor-etal-2019-commonsenseqa,
    title = "{C}ommonsense{QA}: A Question Answering Challenge Targeting Commonsense Knowledge",
    author = "Talmor, Alon  and
      Herzig, Jonathan  and
      Lourie, Nicholas  and
      Berant, Jonathan",
    editor = "Burstein, Jill  and
      Doran, Christy  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1421",
    doi = "10.18653/v1/N19-1421",
    pages = "4149--4158"
}
"""


class CommonsenseQAStd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "commonsense_qa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs: Optional[list] = list(
                map(self._process_doc, self.dataset["train"])
            )
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return NotImplemented

    def _process_doc(self, doc):
        # Question: Sammy wanted to go to where the people were.  Where might he go?
        # Answer: populated areas
        out_doc = {
            "id": doc["id"],
            "query": make_cloze_prompt(doc["question"]),
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

    def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs: Optional[list] = list(
                map(self._process_doc, STD_FEWSHOT[self.DATASET_PATH])
            )
        assert k <= len(self._fewshot_docs)
        return self._fewshot_docs[:k]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class CommonsenseQAMCStd(CommonsenseQAStd):
    # Include answer choices in prompt, answer is just the single letter A, B, ... E.g.,
    # Question: Sammy wanted to go to where the people were.  Where might he go?
    #  A. race track
    #  B. populated areas
    #  C. the desert
    #  D. apartment
    #  E. roadblock
    # Answer: B

    def _process_doc(self, doc):
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
