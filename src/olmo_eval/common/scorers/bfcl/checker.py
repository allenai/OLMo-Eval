"""Compare decoded function calls against BFCL's possible answers.

A possible answer gives, for every parameter, the list of values that count as
correct — so ``{"unit": ["units", ""]}`` accepts either the string ``units`` or
the parameter being left out entirely. Checking a call therefore means matching
its name, confirming every required parameter is present and every supplied
parameter is declared, coercing each value to the type the function document
declares, and finally looking the value up among the accepted ones.

The rules, including the lenient string comparison and the one-level-deep
nested type check, mirror the reference implementation so that a score computed
here is comparable with a published leaderboard number.
"""

from __future__ import annotations

import re
from typing import Any

from .constants import (
    JAVA_TYPE_CONVERSION,
    JS_TYPE_CONVERSION,
    NESTED_CONVERSION_TYPE_LIST,
    PYTHON_NESTED_TYPE_CHECK_LIST,
    PYTHON_TYPE_MAPPING,
    Language,
)

#: Characters dropped before comparing strings, so that "April 1, 2024" and
#: "April 1 2024" are not scored as different answers.
_STANDARDIZE_RE = re.compile(r"[ \,\.\/\-\_\*\^]")

CheckResult = dict[str, Any]

#: Each value checker is reached only when the function document declares the
#: matching type, so the value it is handed is typed as it arrived.


def _ok() -> CheckResult:
    return {"valid": True, "error": []}


def _fail(errors: list[Any], error_type: str) -> CheckResult:
    return {"valid": False, "error": errors, "error_type": error_type}


def standardize_string(input_string: str) -> str:
    """Normalize a string for comparison: drop punctuation, lowercase, requote."""
    return _STANDARDIZE_RE.sub("", input_string).lower().replace("'", '"')


def find_description(func_descriptions: Any, name: str) -> dict[str, Any] | None:
    """Return the function document named ``name``."""
    if isinstance(func_descriptions, list):
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    return func_descriptions


def get_possible_answer_type(possible_answer: list[Any]) -> type | None:
    """Return the type of the first non-omitted accepted value, if any."""
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def type_checker(
    param: str,
    value: Any,
    possible_answer: list[Any],
    expected_type_description: str,
    expected_type_converted: type | None,
    nested_type_converted: type | None,
) -> CheckResult:
    """Check one value against its declared type, one level deep.

    A value whose type matches the *possible answer* rather than the function
    document is treated as a variable name the model wrote instead of a
    literal, and is checked as a string later rather than rejected here.
    """
    result: CheckResult = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }

    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and possible_answer_type is not expected_type_converted:
        is_variable = True

    if type(value) is expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        for possible_answer_item in possible_answer:
            flag = True
            if isinstance(possible_answer_item, list):
                for value_item in value:
                    checker_result = type_checker(
                        param,
                        value_item,
                        possible_answer_item,
                        str(nested_type_converted),
                        nested_type_converted,
                        None,
                    )
                    if not checker_result["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "error": [], "is_variable": is_variable}

        return {
            "valid": False,
            "error": [
                f"Nested type checking failed for parameter {param!r}. Expected outer type "
                f"{expected_type_description} with inner type {nested_type_converted}. "
                f"Parameter value: {value!r}."
            ],
            "error_type": "type_error:nested",
            "is_variable": is_variable,
        }

    if possible_answer_type is not None and type(value) is possible_answer_type:
        result["is_variable"] = True
        return result

    return {
        "valid": False,
        "error": [
            f"Incorrect type for parameter {param!r}. Expected type "
            f"{expected_type_description}, got {type(value).__name__}. "
            f"Parameter value: {value!r}."
        ],
        "error_type": "type_error:simple",
        "is_variable": False,
    }


def string_checker(param: str, model_output: Any, possible_answer: list[Any]) -> CheckResult:
    """Compare a string against the accepted values, ignoring case and spacing."""
    standardized = standardize_string(model_output)
    accepted = [standardize_string(answer) for answer in possible_answer if isinstance(answer, str)]
    if standardized not in accepted:
        return _fail(
            [
                f"Invalid value for parameter {param!r}: {model_output!r}. "
                f"Expected one of {possible_answer}. Case insensitive."
            ],
            "value_error:string",
        )
    return _ok()


def list_checker(param: str, model_output: Any, possible_answer: list[Any]) -> CheckResult:
    """Compare a list against the accepted lists, standardizing string elements."""
    standardized = [
        standardize_string(item) if isinstance(item, str) else item for item in model_output
    ]
    accepted = [
        [standardize_string(item) if isinstance(item, str) else item for item in answer]
        for answer in possible_answer
    ]
    if standardized not in accepted:
        return _fail(
            [
                f"Invalid value for parameter {param!r}: {model_output!r}. "
                f"Expected one of {possible_answer}."
            ],
            "value_error:list/tuple",
        )
    return _ok()


def dict_checker(param: str, model_output: Any, possible_answers: list[Any]) -> CheckResult:
    """Compare a flat dictionary against each accepted dictionary in turn."""
    result: CheckResult = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}

    for possible_answer in possible_answers:
        if possible_answer == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True

        for key, value in model_output.items():
            if key not in possible_answer:
                result = _fail([f"Unexpected dict key parameter: '{key}'."], "value_error:dict_key")
                flag = False
                break

            standardized_value = standardize_string(value) if isinstance(value, str) else value
            accepted = [
                standardize_string(item) if isinstance(item, str) else item
                for item in possible_answer[key]
            ]
            if standardized_value not in accepted:
                result = _fail(
                    [
                        f"Invalid value for parameter {key!r}: {value!r}. "
                        f"Expected one of {accepted}."
                    ],
                    "value_error:dict_value",
                )
                flag = False
                break

        if flag:
            for key, value in possible_answer.items():
                if key not in model_output and "" not in value:
                    result = _fail(
                        [f"Missing dict key parameter: '{key}'."], "value_error:dict_key"
                    )
                    flag = False
                    break

        if flag:
            return _ok()

    return result


def list_dict_checker(param: str, model_output: Any, possible_answers: list[Any]) -> CheckResult:
    """Compare a list of dictionaries, in order, against each accepted list."""
    result: CheckResult = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

    for possible_answer in possible_answers:
        if len(model_output) != len(possible_answer):
            result = _fail(
                ["Wrong number of dictionaries in the list."], "value_error:list_dict_count"
            )
            continue

        flag = True
        for index in range(len(model_output)):
            result = dict_checker(param, model_output[index], [possible_answer[index]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return _ok()

    return result


def _convert_language_value(
    param: str,
    value: Any,
    param_details: dict[str, Any],
    expected_type_description: str,
    language: Language,
) -> tuple[Any, type | None, type | None, CheckResult | None]:
    """Convert a Java or JavaScript argument's source text to a Python value."""
    from .java_js import java_type_converter, js_type_converter

    if language is Language.JAVA:
        conversion_table, convert = JAVA_TYPE_CONVERSION, java_type_converter
        error_type = "type_error:java"
        type_name = "String"
    else:
        conversion_table, convert = JS_TYPE_CONVERSION, js_type_converter
        error_type = "type_error:js"
        type_name = "String"

    expected_type_converted = conversion_table[expected_type_description]
    nested_type_converted: type | None = None

    if not isinstance(value, str):
        return (
            value,
            expected_type_converted,
            None,
            _fail(
                [
                    f"Incorrect type for parameter {param!r}. Expected type {type_name}, "
                    f"got {type(value).__name__}. Parameter value: {value!r}."
                ],
                error_type,
            ),
        )

    if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
        nested_type = param_details[param]["items"]["type"]
        nested_type_converted = conversion_table[nested_type]
        value = convert(value, expected_type_description, nested_type)
    else:
        value = convert(value, expected_type_description)

    return value, expected_type_converted, nested_type_converted, None


def simple_function_checker(
    func_description: dict[str, Any],
    model_output: dict[str, Any],
    possible_answer: dict[str, Any],
    language: Language = Language.PYTHON,
) -> CheckResult:
    """Check one predicted call against one possible answer."""
    accepted_params = next(iter(possible_answer.values()))
    if not func_description:
        return _fail(
            [f"No function document for {next(iter(possible_answer))!r}."],
            "simple_function_checker:wrong_func_name",
        )
    func_name = func_description["name"]
    parameters = func_description.get("parameters") or {}
    param_details = parameters.get("properties") or {}
    required_params = parameters.get("required") or []

    if func_name not in model_output:
        return _fail(
            [f"Function name {func_name!r} not found in model output."],
            "simple_function_checker:wrong_func_name",
        )

    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            return _fail(
                [f"Missing required parameter: {param!r}."],
                "simple_function_checker:missing_required",
            )

    for param, raw_value in model_params.items():
        # Declared loose because the conversions below deliberately change what
        # a value is, guided by the type the function document declares.
        value: Any = raw_value
        if param not in param_details or param not in accepted_params:
            return _fail(
                [f"Unexpected parameter: {param!r}."],
                "simple_function_checker:unexpected_param",
            )

        expected_type_description = param_details[param]["type"]
        nested_type_converted: type | None = None

        if language in (Language.JAVA, Language.JAVASCRIPT):
            (
                value,
                expected_type_converted,
                nested_type_converted,
                conversion_error,
            ) = _convert_language_value(
                param, value, param_details, expected_type_description, language
            )
            if conversion_error is not None:
                return conversion_error
        else:
            expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
            if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
                nested_type_converted = PYTHON_TYPE_MAPPING[param_details[param]["items"]["type"]]

        # A tuple in a possible answer became a list on its way through JSON, so
        # a predicted tuple is compared as one.
        if expected_type_description == "tuple" and isinstance(value, tuple):
            value = list(value)

        # Python widens an int to a float on the way into a call, so a
        # parameter declared float accepts one.
        if (
            language is Language.PYTHON
            and expected_type_description == "float"
            and type(value) is int
        ):
            value = float(value)

        type_check_result = type_checker(
            param,
            value,
            accepted_params[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        if not type_check_result["valid"]:
            return type_check_result
        is_variable = type_check_result["is_variable"]

        # A variable name stands in for a value of any type, so it goes through
        # the plain string comparison below rather than the per-type checks.
        if not is_variable:
            if expected_type_converted is dict:
                result = dict_checker(param, value, accepted_params[param])
                if not result["valid"]:
                    return result
                continue
            if expected_type_converted is list and nested_type_converted is dict:
                result = list_dict_checker(param, value, accepted_params[param])
                if not result["valid"]:
                    return result
                continue
            if expected_type_converted is str:
                result = string_checker(param, value, accepted_params[param])
                if not result["valid"]:
                    return result
                continue
            if expected_type_converted is list:
                result = list_checker(param, value, accepted_params[param])
                if not result["valid"]:
                    return result
                continue

        if value not in accepted_params[param]:
            return _fail(
                [
                    f"Invalid value for parameter {param!r}: {value!r}. "
                    f"Expected one of {accepted_params[param]}."
                ],
                "value_error:others",
            )

    for param in accepted_params:
        if param not in model_params and "" not in accepted_params[param]:
            return _fail(
                [f"Optional parameter {param!r} not provided and not marked as optional."],
                "simple_function_checker:missing_optional",
            )

    return _ok()


def parallel_function_checker_no_order(
    func_descriptions: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
    language: Language = Language.PYTHON,
) -> CheckResult:
    """Check several predicted calls against several answers, in any order."""
    if len(model_output) != len(possible_answers):
        return _fail(
            ["Wrong number of functions."], "parallel_function_checker_no_order:wrong_count"
        )

    matched_indices: list[int] = []

    for answer_index, possible_answer in enumerate(possible_answers):
        func_name_expected = next(iter(possible_answer.keys()))
        func_description = find_description(func_descriptions, func_name_expected)
        all_errors: list[Any] = []
        result: CheckResult = _fail(
            ["No model output left to match."],
            "parallel_function_checker_no_order:cannot_find_match",
        )

        for index in range(len(model_output)):
            if index in matched_indices:
                continue

            result = simple_function_checker(
                func_description or {}, model_output[index], possible_answer, language
            )
            if result["valid"]:
                matched_indices.append(index)
                break
            all_errors.append(
                {
                    f"Model Result Index {index}": {
                        "sub_error": result["error"],
                        "sub_error_type": result["error_type"],
                        "model_output_item": model_output[index],
                        "possible_answer_item": possible_answer,
                    }
                }
            )

        if not result["valid"]:
            considered = [i for i in range(len(model_output)) if i not in matched_indices]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered} of model "
                f"output for index {answer_index} of possible answers.",
            )
            return _fail(all_errors, "parallel_function_checker_no_order:cannot_find_match")

    return _ok()


def multiple_function_checker(
    func_descriptions: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
    language: Language = Language.PYTHON,
) -> CheckResult:
    """Check that the one call picked out of several candidate functions is right."""
    if len(model_output) != len(possible_answers):
        return _fail(["Wrong number of functions."], "multiple_function_checker:wrong_count")

    func_name_expected = next(iter(possible_answers[0].keys()))
    func_description = find_description(func_descriptions, func_name_expected)
    return simple_function_checker(
        func_description or {}, model_output[0], possible_answers[0], language
    )


def ast_checker(
    func_description: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    language: Language,
    test_category: str,
) -> CheckResult:
    """Check a decoded reply against a test entry's possible answers."""
    if "parallel" in test_category:
        return parallel_function_checker_no_order(
            func_description, model_output, possible_answer, language
        )

    if "multiple" in test_category:
        return multiple_function_checker(func_description, model_output, possible_answer, language)

    if len(model_output) != 1:
        return _fail(["Wrong number of functions."], "simple_function_checker:wrong_count")

    return simple_function_checker(
        func_description[0], model_output[0], possible_answer[0], language
    )
