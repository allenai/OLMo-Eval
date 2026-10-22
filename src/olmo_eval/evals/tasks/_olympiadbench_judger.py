"""OlympiadBench's rule-based answer judge, ported for SciReas-Pro scoring.

Ported from ``MathJudger`` in OlympiadBench's ``math_judger.py`` (Apache-2.0), as
vendored by the SciReas evaluation harness (``oe_eval/dependencies/olympiadbench/
code/math_judger.py`` in ``lihaoxin2020/SciReas-Eval`` at revision
``4aca9412f760711adbde24221395525ee6e058bd``). The comparison logic is unchanged.
Differences from the upstream file, none of which changes a verdict:

- bare ``except:`` clauses catch ``Exception``, so an interrupt is no longer
  swallowed;
- the module no longer raises the interpreter's integer string-conversion limit or
  the ``decimal`` precision at import, and logs instead of printing;
- comments are translated to English and the upstream ``__main__`` demo is dropped.

``judge`` takes the first argument as the side whose values get the percentage
variants (``x / 100`` and ``x * 100``) in :meth:`MathJudger.numerical_equal`. The
SciReas harness passes the model answer first, and callers here keep that order.
"""

from __future__ import annotations

import logging
import math
import re

import sympy as sp
from sympy import Eq, Pow, simplify, sympify
from sympy.parsing.latex import parse_latex

logger = logging.getLogger(__name__)

DEFAULT_PRECISION = 1e-8


class MathJudger:
    """Rule-based equivalence of two LaTeX answers: numeric, symbolic, equation, interval."""

    def __init__(self) -> None:
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = DEFAULT_PRECISION

    def split_by_comma(self, expr: str) -> list[str]:
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char == "(" or char == "[":
                in_bracket_num += 1
            elif char == ")" or char == "]":
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list[str]) -> list[str]:
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(
        self,
        expression1: str,
        expression2: str,
        precision: float | list[float] = DEFAULT_PRECISION,
    ) -> bool:
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except Exception:
            return False
        if expression1 == expression2:
            return True

        # Drop CJK characters; answers made only of them were compared verbatim above.
        expression1 = re.sub(r"[\u4e00-\u9fff]+", "", expression1)
        expression2 = re.sub(r"[\u4e00-\u9fff]+", "", expression2)

        temp_list1 = self.trans_plus_minus_sign(self.split_by_comma(expression1))
        temp_list2 = self.trans_plus_minus_sign(self.split_by_comma(expression2))

        # One tolerance per answer when a single one is given.
        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # Every answer must pair with an equal answer on the other side.
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                return False

        return True

    def is_interval(self, epr: str) -> bool:
        return epr.startswith(("(", "[")) and epr.endswith((")", "]"))

    def sympy_sub_pi(self, expression_sympy):
        """Replace the symbol pi with its value before numeric comparison."""
        return expression_sympy.subs(self.pi, math.pi)

    def is_equal(self, expression1: str, expression2: str) -> bool:
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            return True

        # Two intervals are compared as intervals and nothing else.
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    return True
            except Exception:
                return False

        try:
            if self.numerical_equal(expression1, expression2):
                return True
        except Exception:
            pass

        try:
            if self.expression_equal(expression1, expression2) and not (
                "=" in expression1 and "=" in expression2
            ):
                return True
        except Exception:
            pass

        try:
            if self.equation_equal(expression1, expression2):
                return True
        except Exception:
            pass

        return False

    def numerical_equal(
        self, expression1: str, expression2: str, include_percentage: bool = True
    ) -> bool:
        """Whether two numbers agree within ``self.precision``, allowing a percent scale."""
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        return any(abs(item - prediction) <= self.precision * 1.01 for item in gt_result)

    def expression_equal(self, exp1: str, exp2: str) -> bool:
        """Whether two expressions are mathematically equivalent (right of any ``=``)."""

        def extract_expression(expression: str) -> str:
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        expr1_sym = self.sympy_sub_pi(expr1_sym)
        expr2_sym = self.sympy_sub_pi(expr2_sym)

        if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
            not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)
        ):
            return False
        if not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
            try:
                if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                    logger.debug(
                        "Cannot evaluate %s and %s numerically; treating as unequal",
                        expr1_sym,
                        expr2_sym,
                    )
                    return False
                return bool(abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01)
            except Exception:
                return False
        try:
            simplified_expr = simplify(expr1_sym - expr2_sym)
            num_value = simplified_expr.evalf()
            return bool(abs(num_value) < 1e-3)
        except Exception:
            return False

    def equation_equal(self, expression1: str, expression2: str) -> bool:
        """Whether two equations are equivalent: their zero forms differ by an integer factor."""

        def simplify_equation(latex_eq: str):
            lhs, rhs = latex_eq.split("=")
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)
            equation = Eq(lhs_expr, rhs_expr)
            return simplify(equation.lhs - equation.rhs)

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        return bool(
            (division_result_1.is_Integer and division_result_1 != 0)
            or (division_result_2.is_Integer and division_result_2 != 0)
        )

    def interval_equal(self, expression1: str, expression2: str) -> bool:
        """Whether two (unions of) intervals have the same brackets and equal endpoints."""

        def compare_two_interval(inter1: str, inter2: str) -> bool:
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            items_1 = inter1.strip("[]()").split(",")
            items_2 = inter2.strip("[]()").split(",")

            for item_1, item_2 in zip(items_1, items_2, strict=False):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        if expression1 == expression2:
            return True
        inter_list1 = expression1.split("\\cup")
        inter_list2 = expression2.split("\\cup")
        if len(inter_list1) != len(inter_list2):
            return False
        for inter1, inter2 in zip(inter_list1, inter_list2, strict=True):
            if not compare_two_interval(inter1, inter2):
                return False
        return True

    def preprocess(self, expression1: str, expression2: str) -> tuple[str, str]:
        def extract_boxed_content(latex_str: str) -> str:
            # Every \boxed{...} joined by commas; without one, the $...$ formulas of
            # the last line, or else the whole string.
            boxed_matches = re.finditer(r"\\boxed{", latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == "{":
                        stack += 1
                    elif latex_str[end_index] == "}":
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index : end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def special_symbol_replace(expression: str) -> str:
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            # Cosmetic symbols that do not change how the LaTeX parses.
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            # A set of characters to trim, as upstream.
            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")  # noqa: B005

            pattern = r"\\(?:mathrm|mathbf)\{~?([^}]*)\}"
            return re.sub(pattern, r"\1", expression)

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        return special_symbol_replace(exp1), special_symbol_replace(exp2)

    def can_compute_power(self, expr) -> bool:
        """Whether a numeric power has an exponent small enough to evaluate."""
        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                max_exp = 1000
                return not abs(exp.evalf()) > max_exp
            return False
        return True
