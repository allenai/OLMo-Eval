"""Java and JavaScript call parsing and value conversion for BFCL.

Two of the AST categories ask for calls written in Java or JavaScript rather
than Python. Their arguments arrive as source text — ``new int[]{1, 2}``,
``42L``, ``{a: 1}`` — so each value is converted to the Python value the
checker compares before it is compared.

The grammars come from ``tree-sitter-java`` and ``tree-sitter-javascript``,
which the java and javascript tasks declare as runtime dependencies; a task
that never sees those categories never needs them installed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from .constants import JAVA_TYPE_CONVERSION, JS_TYPE_CONVERSION

_GRAMMAR_PACKAGES = {
    "java": "tree-sitter-java",
    "javascript": "tree-sitter-javascript",
}


@lru_cache(maxsize=2)
def _parser(language: str) -> Any:
    """Return a tree-sitter parser for one language, built once per process."""
    try:
        from tree_sitter import Language, Parser

        if language == "java":
            import tree_sitter_java as grammar  # type: ignore[ty:unresolved-import]
        else:
            import tree_sitter_javascript as grammar  # type: ignore[ty:unresolved-import]
    except ImportError as exc:
        package = _GRAMMAR_PACKAGES[language]
        raise ImportError(
            f"Parsing {language} function calls needs the {package} grammar. "
            f"Install it with `uv pip install '{package}>=0.23,<0.24'`, or run the "
            f"task through a runner that installs its declared dependencies."
        ) from exc

    return Parser(Language(grammar.language()))


def _parse_tree(source_code: str, language: str) -> Any:
    """Parse source text, raising if the grammar could not make sense of it."""
    tree = _parser(language).parse(source_code.encode("utf-8"))
    if "ERROR" in str(tree.root_node):
        raise SyntaxError(f"Could not parse the {language} source code.")
    return tree.root_node


def parse_java_function_call(source_code: str) -> list[dict[str, Any]]:
    """Parse one Java method invocation into ``[{name: {param: value}}]``."""
    root_node = _parse_tree(source_code, "java")

    def get_text(node: Any) -> str:
        return source_code[node.start_byte : node.end_byte]

    def traverse_node(node: Any, nested: bool = False) -> str:
        if node.type == "string_literal":
            # A nested literal keeps its quotes so the value converter can tell
            # a string element from a bare identifier.
            return get_text(node) if nested else get_text(node)[1:-1]
        if node.type == "character_literal":
            return get_text(node) if nested else get_text(node)[1:-1]
        if node.type in ("identifier", "class_literal", "type_identifier", "method_invocation"):
            return get_text(node)
        if node.type == "array_creation_expression":
            type_text = traverse_node(node.child_by_field_name("type"), True)
            value_text = traverse_node(node.child_by_field_name("value"), True)
            return f"new {type_text}[]{value_text}"
        if node.type == "object_creation_expression":
            type_text = traverse_node(node.child_by_field_name("type"), True)
            arguments_node = node.child_by_field_name("arguments")
            if arguments_node is None:
                return f"new {type_text}()"
            argument_texts = [
                traverse_node(child, True)
                for child in arguments_node.children
                if child.type not in (",", "(", ")")
            ]
            return f"new {type_text}({', '.join(argument_texts)})"
        if node.type == "set":
            items = [
                traverse_node(child, True)
                for child in node.children
                if child.type not in (",", "set")
            ]
            return "{" + ", ".join(items) + "}"
        if node.child_count > 0:
            return "".join(traverse_node(child, True) for child in node.children)
        return get_text(node)

    def extract_arguments(args_node: Any) -> dict[Any, Any]:
        arguments: dict[Any, Any] = {}

        def record(name: Any, value: str) -> None:
            if name in arguments:
                if not isinstance(arguments[name], list):
                    arguments[name] = [arguments[name]]
                arguments[name].append(value)
            else:
                arguments[name] = value

        for child in args_node.children:
            if child.type == "assignment_expression":
                name_node, value_node = child.children[0], child.children[2]
                record(get_text(name_node), traverse_node(value_node))
            elif child.type in ("identifier", "class_literal", "set"):
                record(None, traverse_node(child))
        return arguments

    def traverse(node: Any) -> list[dict[str, Any]] | None:
        if node.type == "method_invocation":
            method_name = get_text(node.child_by_field_name("name"))
            class_name_node = node.child_by_field_name("object")
            function_name = (
                f"{get_text(class_name_node)}.{method_name}" if class_name_node else method_name
            )
            arguments_node = node.child_by_field_name("arguments")
            if arguments_node is None:
                return None
            arguments = extract_arguments(arguments_node)
            if any(isinstance(value, list) for value in arguments.values()):
                raise ValueError("Multiple arguments with the same name are not supported.")
            return [{function_name: arguments}]
        for child in node.children:
            result = traverse(child)
            if result:
                return result
        return None

    return traverse(root_node) or []


def parse_javascript_function_call(source_code: str) -> list[dict[str, Any]]:
    """Parse one JavaScript call expression into ``[{name: {param: value}}]``."""
    root_node = _parse_tree(source_code, "javascript")

    def extract_arguments(node: Any) -> dict[Any, Any]:
        args: dict[Any, Any] = {}

        def record(name: Any, value: str) -> None:
            if name in args:
                if not isinstance(args[name], list):
                    args[name] = [args[name]]
                args[name].append(value)
            else:
                args[name] = value

        for child in node.children:
            if child.type == "assignment_expression":
                name = child.children[0].text.decode("utf-8")
                value = child.children[2].text.decode("utf-8")
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                record(name, value)
            elif child.type in ("identifier", "true"):
                record(None, child.text.decode("utf-8"))
        return args

    if root_node.type != "program":
        return []
    for child in root_node.children:
        if child.type != "expression_statement":
            continue
        for sub_child in child.children:
            if sub_child.type != "call_expression":
                continue
            function_name = sub_child.children[0].text.decode("utf-8")
            parameters = extract_arguments(sub_child.children[1])
            if any(isinstance(value, list) for value in parameters.values()):
                raise ValueError("Multiple arguments with the same name are not supported.")
            return [{function_name: parameters}]
    return []


def java_type_converter(value: str, expected_type: str, nested_type: str | None = None) -> Any:
    """Convert Java source text to the Python value the checker compares.

    Text that does not match the declared type is passed through as a string,
    so a mismatch surfaces as a value error against the possible answers rather
    than as a decoding failure.
    """
    if expected_type not in JAVA_TYPE_CONVERSION:
        raise ValueError(f"Unsupported type: {expected_type}")

    if expected_type in ("byte", "short", "integer"):
        return int(value) if re.match(r"^-?\d+$", value) else str(value)
    if expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value):
            return str(value)
        return float(re.sub(r"[fF]$", "", value))
    if expected_type == "double":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", value):
            return str(value)
        return float(value)
    if expected_type == "long":
        if not re.match(r"^-?\d+[lL]$", value):
            return str(value)
        return int(re.sub(r"[lL]$", "", value))
    if expected_type == "boolean":
        return value == "true" if value in ("true", "false") else str(value)
    if expected_type == "char":
        return value if re.match(r"^'.'$", value) else str(value)
    if expected_type == "ArrayList":
        return _parse_java_arraylist(value, nested_type)
    if expected_type == "Array":
        return _parse_java_array(value, nested_type)
    if expected_type == "HashMap":
        return _parse_java_hashmap(value)
    if expected_type in ("Set", "Hashtable", "Queue", "Stack"):
        raise NotImplementedError(f"{expected_type} conversion is not implemented")
    # String and any: `any` is compared as its string representation.
    return str(value)


def _convert_java_element(element: str, nested_type: str | None) -> Any:
    """Convert one collection element, honoring the declared element type."""
    if nested_type in ("char", "String"):
        return element[1:-1]
    if nested_type:
        return java_type_converter(element, nested_type)
    return _parse_java_value(element)


def _parse_java_arraylist(input_str: str, nested_type: str | None = None) -> Any:
    match_as_list = re.search(r"new\s+ArrayList<\w*>\(Arrays\.asList\((.+?)\)\)", input_str)
    if match_as_list:
        return [
            _convert_java_element(element.strip(), nested_type)
            for element in match_as_list.group(1).split(",")
        ]

    match_add = re.search(r"new\s+ArrayList<\w*>\(\)\s*\{\{\s*(.+?)\s*\}\}", input_str, re.DOTALL)
    if match_add:
        return [
            _convert_java_element(match.strip(), nested_type)
            for match in re.findall(r"add\((.+?)\)", match_add.group(1))
        ]

    if re.search(r"new\s+ArrayList<\w*>\(\)", input_str):
        return []
    return input_str


def _parse_java_array(input_str: str, nested_type: str | None = None) -> Any:
    match = re.search(r"new\s+\w+\[\]\s*\{(.*?)\}", input_str)
    if not match:
        return input_str
    elements = [element.strip() for element in match.group(1).split(",") if element.strip()]
    if nested_type:
        return [java_type_converter(element, nested_type) for element in elements]
    return [_parse_java_value(element) for element in elements]


def _parse_java_hashmap(input_str: str) -> Any:
    match = re.search(
        r"new\s+HashMap<.*?>\s*\(\)\s*\{\s*\{?\s*(.*?)\s*\}?\s*\}", input_str, re.DOTALL
    )
    if match:
        entries: dict[str, Any] = {}
        if match.group(1).strip():
            for key, value in re.findall(r'put\("(.*?)",\s*(.*?)\)', match.group(1)):
                entries[key] = _parse_java_value(value.strip())
        return entries

    if re.search(r"new\s+HashMap<.*?>\s*\(\)", input_str):
        return {}
    return input_str


def _parse_java_value(value_str: str) -> Any:
    """Convert Java source text with no declared type to a Python value."""
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    if re.match(r"^-?\d+[lL]$", value_str):
        return int(value_str[:-1])
    if re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value_str):
        return float(re.sub(r"[fF]$", "", value_str))
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        return value_str


def js_type_converter(value: str, expected_type: str, nested_type: str | None = None) -> Any:
    """Convert JavaScript source text to the Python value the checker compares."""
    if expected_type not in JS_TYPE_CONVERSION:
        raise ValueError(f"Unsupported type: {expected_type}")

    if expected_type == "String":
        quoted = (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        )
        return value[1:-1] if quoted else str(value)
    if expected_type == "integer":
        return int(value) if re.match(r"^-?\d+$", value) else str(value)
    if expected_type == "float":
        return float(value) if re.match(r"^-?\d+(\.\d+)?$", value) else str(value)
    if expected_type == "Bigint":
        return int(value[:-1]) if re.match(r"^-?\d+n$", value) else str(value)
    if expected_type == "Boolean":
        return value == "true" if value in ("true", "false") else str(value)
    if expected_type in ("dict", "array"):
        return _parse_js_collection(value, expected_type, nested_type)
    return str(value)


_JS_ARRAY_2D_PATTERN = (
    r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]|\bnew\s+Array\(\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\)"
)
_JS_ARRAY_PATTERN = r"\[(.*?)\]|\bnew\s+Array\((.*?)\)"


def _parse_js_collection(code: str, type_str: str, nested_type: str | None = None) -> Any:
    code = code.strip()
    if type_str == "array":
        return _parse_js_array(code, nested_type)
    if code == "{}":
        return {}
    dict_match = re.match(r"\{(.*?)\}", code)
    if not dict_match:
        return code
    try:
        entries: dict[str, Any] = {}
        pairs = re.findall(r"([^:]+):\s*(.*?)(?:,\s*(?=[^,]+:)|$)", dict_match.group(1))
        for key, value in pairs:
            key = key.strip().strip("'\"")
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                entries[key] = _parse_js_collection(value, "array")
            elif value.startswith("{") and value.endswith("}"):
                entries[key] = _parse_js_collection(value, "dict")
            else:
                entries[key] = _parse_js_value(value.strip("'\""))
        return entries
    except Exception:
        return code


def _parse_js_array(code: str, nested_type: str | None) -> Any:
    try:
        array_2d_match = re.match(_JS_ARRAY_2D_PATTERN, code)
        if array_2d_match:
            rows = []
            for index, inner in enumerate(re.findall(r"\[(.*?)\]", array_2d_match.group(0))):
                inner = inner.strip()
                if index == 0 and inner.startswith("["):
                    inner = inner[1:]
                rows.append([_parse_js_value(element.strip()) for element in inner.split(",")])
            return rows

        array_match = re.match(_JS_ARRAY_PATTERN, code)
        if not array_match:
            return code
        elements_str = (array_match.group(1) or array_match.group(2) or "").strip()
        elements = elements_str.split(",") if elements_str else []
        if not nested_type:
            return [_parse_js_value(element.strip()) for element in elements]
        converted = []
        for element in elements:
            element = element.strip()
            quoted = element.startswith("'") or element.startswith('"')
            converted.append(
                js_type_converter(element, nested_type, "String")
                if quoted
                else js_type_converter(element, nested_type)
            )
        return converted
    except Exception:
        return code


def _parse_js_value(value_str: str) -> Any:
    """Convert JavaScript source text with no declared type to a Python value."""
    value_str = value_str.strip()
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        return value_str
