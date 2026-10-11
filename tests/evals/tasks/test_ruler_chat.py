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
