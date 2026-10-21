"""Tests for extract_mcq_answer."""

from olmo_eval.evals.extract.mcq import extract_mcq_answer


class TestExtractMcqAnswer:
    """Tests for the shared MCQ answer extractor."""

    # --- ANSWER: X ---

    def test_answer_colon(self):
        assert extract_mcq_answer("ANSWER: B") == "B"

    def test_answer_colon_lowercase(self):
        assert extract_mcq_answer("answer: c") == "C"

    def test_answer_colon_no_space(self):
        assert extract_mcq_answer("ANSWER:A") == "A"

    def test_answer_colon_with_paren(self):
        assert extract_mcq_answer("ANSWER: (D)") == "D"

    def test_answer_colon_last_wins(self):
        assert extract_mcq_answer("ANSWER: A\nActually ANSWER: C") == "C"

    def test_answer_bold_label(self):
        assert extract_mcq_answer("**Answer:** B") == "B"

    def test_answer_bold_letter(self):
        assert extract_mcq_answer("Answer: **(C)**") == "C"

    def test_answer_bold_wraps_label_and_letter(self):
        assert extract_mcq_answer("**ANSWER: D**") == "D"

    def test_answer_underscore_emphasis(self):
        assert extract_mcq_answer("Answer: _B_") == "B"
        assert extract_mcq_answer("Answer: __C__") == "C"

    def test_answer_bold_label_closed_before_colon(self):
        assert extract_mcq_answer("**Answer**: B") == "B"

    def test_answer_letter_on_next_line(self):
        assert extract_mcq_answer("ANSWER:\nC") == "C"
        assert extract_mcq_answer("**Answer:**\n\n**B**") == "B"
        assert extract_mcq_answer("Answer:\n  D.") == "D"

    def test_answer_option_label_on_next_line(self):
        assert extract_mcq_answer("Answer:\nB) Paris") == "B"
        assert extract_mcq_answer("Answer:\n(C) Paris") == "C"

    def test_answer_sentence_on_next_line_does_not_capture_word(self):
        assert extract_mcq_answer("Answer:\n\nThe correct choice is B.") is None
        assert extract_mcq_answer("Answer:\nA good approach is to compare options") is None
        assert extract_mcq_answer("Answer:\nI think so") is None

    def test_answer_colon_followed_by_word_does_not_capture_first_letter(self):
        """'ANSWER: the ...' must not capture T from 'the'."""
        assert extract_mcq_answer("ANSWER: the correct choice is C") is None

    def test_answer_colon_followed_by_word_falls_back_to_later_pattern(self):
        assert extract_mcq_answer("ANSWER: the correct choice is (C)") == "C"

    # --- ANSWER X (no colon) ---

    def test_answer_without_colon(self):
        assert extract_mcq_answer("Answer C") == "C"
        assert extract_mcq_answer("ANSWER B") == "B"
        assert extract_mcq_answer("Final Answer D") == "D"

    def test_answer_without_colon_with_markup(self):
        assert extract_mcq_answer("**Answer** B") == "B"
        assert extract_mcq_answer("**Answer C**") == "C"
        assert extract_mcq_answer("Answer (C)") == "C"
        assert extract_mcq_answer("Answer _D_") == "D"
        assert extract_mcq_answer("**Final Answer** (B) mitochondria") == "B"

    def test_answer_without_colon_option_label(self):
        assert extract_mcq_answer("Answer C) Paris") == "C"
        assert extract_mcq_answer("Answer C.") == "C"

    def test_answer_without_colon_last_wins(self):
        assert extract_mcq_answer("Answer A\nWait, no.\nAnswer C") == "C"

    def test_answer_without_colon_preferred_over_boxed(self):
        assert extract_mcq_answer("\\boxed{A}\nAnswer B") == "B"

    def test_answer_without_colon_ignores_lowercase_letter(self):
        """Without a colon, 'answer a) first' is prose, not an answer."""
        assert extract_mcq_answer("Let me answer a) first.\n\\boxed{C}") == "C"
        assert extract_mcq_answer("answer b\n\\boxed{D}") == "D"

    def test_answer_without_colon_ignores_sentence_start(self):
        """Without a colon, the letter must stand alone or label an option."""
        assert extract_mcq_answer("To answer A good question needs care.\n\\boxed{C}") == "C"
        assert extract_mcq_answer("I'll answer I think\n\\boxed{B}") == "B"
        assert extract_mcq_answer("Answer C is correct") is None
        assert extract_mcq_answer("Answer C, because") is None

    def test_answer_without_colon_stays_on_same_line(self):
        assert extract_mcq_answer("### Answer\n\nC") is None
        assert extract_mcq_answer("Answer\nB") is None

    def test_answer_without_colon_needs_separate_word(self):
        assert extract_mcq_answer("The answer is C") is None
        assert extract_mcq_answer("answers C)") is None
        assert extract_mcq_answer("answered A.") is None
        assert extract_mcq_answer("Answering B.") is None
        assert extract_mcq_answer("Answer AB") is None
        assert extract_mcq_answer("Answer A1") is None

    def test_answer_without_colon_long_emphasis_run(self):
        """A long run of emphasis markers is not an answer."""
        assert extract_mcq_answer("Answer " + "*" * 20000) is None
        assert extract_mcq_answer("Answer " + "_" * 20000 + "B") == "B"

    # --- \boxed{X} ---

    def test_boxed_letter(self):
        assert extract_mcq_answer("$$\\boxed{B}$$") == "B"

    def test_boxed_text(self):
        assert extract_mcq_answer("$$\\boxed{\\text{A}}$$") == "A"

    def test_boxed_last_wins(self):
        assert extract_mcq_answer("\\boxed{A}\n\\boxed{D}") == "D"

    # --- (X) ---

    def test_paren_letter(self):
        assert extract_mcq_answer("**(C) Diabetes**") == "C"

    def test_paren_last_wins(self):
        assert extract_mcq_answer("(A) is wrong\n**Answer: (D)**") == "D"

    # --- Priority order ---

    def test_answer_preferred_over_boxed(self):
        assert extract_mcq_answer("\\boxed{A}\nANSWER: B") == "B"

    def test_boxed_preferred_over_paren(self):
        assert extract_mcq_answer("(A) is likely\n$$\\boxed{C}$$") == "C"

    # --- **X) bold markdown ---

    def test_bold_letter_paren(self):
        assert extract_mcq_answer("### Final Answer\n\n**D) Intubate**") == "D"

    def test_bold_letter_dot(self):
        assert extract_mcq_answer("### Final Answer\n\n**C. Staphylococcus aureus**") == "C"

    def test_paren_preferred_over_bold(self):
        assert extract_mcq_answer("**B) wrong**\n**(C) right**") == "C"

    # --- Answer: must stay on same line ---

    def test_answer_colon_newline_does_not_match_next_word(self):
        """'Answer:\\n\\nThe ...' must not capture T from 'The'."""
        text = "### Final Answer:\n\nThe answer is obvious\n\\boxed{D}"
        assert extract_mcq_answer(text) == "D"

    # --- No match ---

    def test_no_match(self):
        assert extract_mcq_answer("I think the answer is B") is None

    def test_empty(self):
        assert extract_mcq_answer("") is None
