import pytest

from olmo_eval.common.types import Instance, RequestType
from olmo_eval.evals import suites, tasks  # noqa: F401  (registers tasks and suites)
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common.registry import get_task, list_variants


def test_every_ruler_task_has_a_chat_variant():
    variants = list_variants()
    ruler = [t for t in variants if t.startswith("ruler_")]
    assert ruler
    assert all("chat" in variants[t] for t in ruler)


def test_chat_variant_sends_the_completion_prompt_as_one_user_turn():
    completion = get_task("ruler_niah_s_1__65536")
    chat = get_task("ruler_niah_s_1__65536:chat")
    question = "haystack ... The special magic number is:"
    instance = Instance(question=question, gold_answer=["42"], metadata={})
    plain = completion.format_request(instance)
    req = chat.format_request(instance)
    assert plain.request_type == RequestType.COMPLETION
    assert req.request_type == RequestType.CHAT
    assert req.messages == ({"role": "user", "content": plain.prompt},)
    assert chat.config.sampling_params == completion.config.sampling_params


def test_chat_suite_mirrors_the_completion_suite():
    plain, chat = get_suite("ruler_all__65536"), get_suite("ruler_all_chat__65536")
    assert chat.tasks == tuple(f"{t}:chat" for t in plain.tasks)
    assert len(chat.tasks) == 13


@pytest.mark.parametrize("name", ["ruler_qa_1__65536", "ruler_niah_s_1__65536"])
@pytest.mark.parametrize("input_text", [None, ""])
def test_chat_preserves_the_fallback_answer_prefix(name, input_text):
    from olmo_eval.data.ruler_loader import get_ruler_templates

    completion, chat = get_task(name), get_task(f"{name}:chat")
    user, system, prompt = get_ruler_templates(completion.task_type)
    doc = {
        "input": input_text,
        "context": "The value is 42.",
        "question": "Value?",
        "query": "key",
        "type_needle_v": "number",
        "outputs": ["42"],
    }
    for task in (completion, chat):
        task._templates = {"user": user, "system": system, "prompt": prompt}
    plain = completion.format_request(completion.process_doc(doc))
    req = chat.format_request(chat.process_doc(doc))
    assert req.messages == ({"role": "user", "content": plain.prompt},)
    assert chat.request_type == RequestType.CHAT
