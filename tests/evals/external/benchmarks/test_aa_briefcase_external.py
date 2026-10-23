"""Tests for the AA-Briefcase-Lite external evaluation."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.scorers import llm_judge
from olmo_eval.evals.external.benchmarks.aa_briefcase import eval as bc_eval
from olmo_eval.evals.external.benchmarks.aa_briefcase.eval import (
    AABriefcaseLiteExternalEval,
    BriefcaseArgs,
    TrialResult,
    build_agent_prompts,
    compute_metrics,
    deliverable_sources,
)
from olmo_eval.evals.external.benchmarks.aa_briefcase.judge import (
    CheckVerdict,
    ParsedSubmission,
    build_judge_content,
    grade_check,
    parse_verdict,
)
from olmo_eval.evals.external.benchmarks.aa_briefcase.loader import (
    load_scenario,
    select_tasks,
    stage_sources,
)
from olmo_eval.evals.external.registry import get_external_eval

PARSER = Path(bc_eval.SANDBOX_CONTEXT) / "parse_submission.py"

JUDGE_USER = """\
<original_task_instructions>
{task_markdown}
</original_task_instructions>
<rubric_item>
{check_description}|{score_1_criteria}|{score_0_criteria}
</rubric_item>
<submitted_artifact>
<<<SUBMISSION CONTENT MESSAGES>>>
</submitted_artifact>
"""


def check_row(check_id: str, task_id: str, files: str, scoring: str = "binary") -> dict:
    return {
        "check_id": check_id,
        "task_id": task_id,
        "week": "1",
        "check_type": check_id.split("-")[2],
        "scoring_type": scoring,
        "taskdoer_output_file": files,
        "files_needed": [],
        "check_description": f"describe {check_id}",
        "score_1_criteria": f"pass {check_id}",
        "score_0_criteria": f"fail {check_id}",
        "specific_file_locations": "",
    }


@pytest.fixture
def scenario_root(tmp_path: Path) -> Path:
    """A two-task scenario laid out like the dataset."""
    root = tmp_path / "dataset"
    for rel, text in {
        "tasks/w1_t1.md": "# Map the market",
        "tasks/w1_t2.md": "# Model the market",
        "summary_docs/scenario_overview.md": "Scenario overview.",
        "summary_docs/week_1/week_1_overview.md": "Week overview.",
        "prompts/eval_system.txt": "Up to {max_turns} steps; finish with `{finish_tool_name}`, "
        "give up with `{abandon_task_finish}`.",
        "prompts/eval_submission.txt": "Call `{finish_tool_name}`.\n{scenario_overview}\n"
        "{week_overview}\n{task}\n{expected_output_filenames}",
        "prompts/judge_system.txt": "Grade strictly.",
        "prompts/judge_user.txt": JUDGE_USER,
        "source_files/shared/report.pdf": "pdf",
        "source_files/shared/slack/general/day.json": "{}",
        "source_files/week/email.eml": "mail",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    common = {
        "week": 1,
        "scenario_overview_path": "summary_docs/scenario_overview.md",
        "week_overview_path": "summary_docs/week_1/week_1_overview.md",
        "shared_files": ["source_files/shared/report.pdf", "source_files/shared/slack/"],
        "week_files": ["source_files/week/email.eml"],
    }
    tasks = [
        {
            "task_id": "w1_t1",
            "task_md_path": "tasks/w1_t1.md",
            "deliverable_filenames": ["overview.tex", "overview.pdf"],
            **common,
        },
        {
            "task_id": "w1_t2",
            "task_md_path": "tasks/w1_t2.md",
            "deliverable_filenames": ["model.xlsx"],
            **common,
        },
    ]
    checks = [
        check_row("W1-T1-A-01", "w1_t1", "overview.pdf, overview.tex"),
        check_row("W1-T1-C-01", "w1_t1", "overview.tex"),
        check_row("W1-T1-AQ-01", "w1_t1", "overview.tex", scoring="pairwise"),
        check_row("W1-T2-A-01", "w1_t2", "model.xlsx"),
    ]
    (root / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    (root / "checks.jsonl").write_text("\n".join(json.dumps(c) for c in checks))
    return root


def parsed_submission(tmp_path: Path, files: dict[str, list[dict]]) -> ParsedSubmission:
    (tmp_path / "page.png").write_bytes(b"PNG")
    return ParsedSubmission(files=files, image_dir=tmp_path)


def scripted_judge(*replies: str | Exception):
    remaining = list(replies)
    prompts: list[Any] = []

    async def judge(prompt, *, system_prompt=None):
        prompts.append((prompt, system_prompt))
        reply = remaining.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    judge.judge_config = {"model": "scripted"}  # type: ignore[attr-defined]
    return judge, prompts


def verdict(passed: bool) -> str:
    return json.dumps({"reasoning": "because", "passed": passed})


@pytest.fixture
def no_waits(monkeypatch):
    """Judge retries without waiting."""

    async def call_once(func, **_):
        return await func()

    async def no_sleep(_):
        return None

    monkeypatch.setattr(llm_judge, "retry_with_backoff", call_once)
    monkeypatch.setattr(llm_judge.asyncio, "sleep", no_sleep)


class TestLoader:
    def test_reads_tasks_binary_checks_and_prompts(self, scenario_root: Path) -> None:
        scenario = load_scenario(scenario_root)

        assert [t.task_id for t in scenario.tasks] == ["w1_t1", "w1_t2"]
        first = scenario.tasks[0]
        assert first.brief == "# Map the market"
        assert first.deliverables == ("overview.tex", "overview.pdf")
        assert [c.check_id for c in first.checks] == ["W1-T1-A-01", "W1-T1-C-01"]
        assert first.checks[0].files == ("overview.pdf", "overview.tex")
        assert scenario.pairwise_checks == 1
        assert scenario.judge_system_prompt == "Grade strictly."

    def test_select_tasks_keeps_scenario_order_and_rejects_unknown_ids(
        self, scenario_root: Path
    ) -> None:
        tasks = load_scenario(scenario_root).tasks
        assert [t.task_id for t in select_tasks(tasks, ["w1_t2"])] == ["w1_t2"]
        assert len(select_tasks(tasks, None)) == 2
        with pytest.raises(ValueError, match="w9_t9"):
            select_tasks(tasks, ["w9_t9"])

    def test_stage_sources_copies_files_and_directories(
        self, scenario_root: Path, tmp_path: Path
    ) -> None:
        task = load_scenario(scenario_root).tasks[0]
        stage_sources(scenario_root, task, tmp_path / "staged")

        assert (tmp_path / "staged/shared/report.pdf").read_text() == "pdf"
        assert (tmp_path / "staged/shared/slack/general/day.json").is_file()
        assert (tmp_path / "staged/week/email.eml").read_text() == "mail"


class TestPrompts:
    def test_agent_prompts_fill_every_placeholder(self, scenario_root: Path) -> None:
        scenario = load_scenario(scenario_root)
        system, instruction = build_agent_prompts(scenario, scenario.tasks[0], 500)

        assert system == (
            "Up to 500 steps; finish with `finish`, give up with `abandon_task_finish`."
        )
        assert instruction == (
            "Call `finish`.\nScenario overview.\nWeek overview.\n# Map the market\n"
            "- overview.tex\n- overview.pdf"
        )

    def test_deliverables_are_matched_by_file_name(self) -> None:
        sources = deliverable_sources(
            ("a.pdf", "b.xlsx"), ["/home/user/a.pdf", "/home/user/work/b.xlsx", "/tmp/c.txt"]
        )
        assert sources == {"a.pdf": "/home/user/a.pdf", "b.xlsx": "/home/user/work/b.xlsx"}

    def test_judge_content_inlines_the_deliverables_at_the_marker(
        self, scenario_root: Path, tmp_path: Path
    ) -> None:
        scenario = load_scenario(scenario_root)
        task = scenario.tasks[0]
        submission = parsed_submission(
            tmp_path,
            {
                "overview.pdf": [
                    {"type": "text", "text": "page text"},
                    {"type": "image", "path": "page.png"},
                    {"type": "note", "text": "only 1 page shown"},
                ]
            },
        )
        parts = build_judge_content(scenario, task.brief, task.checks[0], submission)

        assert parts[0]["text"].startswith("<original_task_instructions>\n# Map the market")
        assert "describe W1-T1-A-01|pass W1-T1-A-01|fail W1-T1-A-01" in parts[0]["text"]
        assert parts[1:] == [
            {"type": "text", "text": "File: overview.pdf"},
            {"type": "text", "text": "page text"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"PNG").decode()},
            },
            {"type": "text", "text": "[Parser note: only 1 page shown]"},
            {"type": "text", "text": "File: overview.tex"},
            {"type": "text", "text": "[Parser note: overview.tex was not submitted.]"},
            {"type": "text", "text": "\n</submitted_artifact>\n"},
        ]

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            (verdict(True), (True, "because")),
            ('{"passed": false}', (False, "")),
            ('{"passed": "yes"}', None),
            ("PASS", None),
            ("[]", None),
        ],
    )
    def test_parse_verdict(self, reply: str, expected) -> None:
        assert parse_verdict(reply) == expected


class TestGradeCheck:
    def _grade(self, scenario_root: Path, judge, submission, check_index: int = 0):
        scenario = load_scenario(scenario_root)
        task = scenario.tasks[0]
        return asyncio.run(
            grade_check(
                judge,
                scenario,
                task.brief,
                task.checks[check_index],
                submission,
                asyncio.Semaphore(1),
            )
        )

    def test_a_check_with_no_submitted_file_fails_without_a_call(
        self, scenario_root: Path, tmp_path: Path
    ) -> None:
        judge, prompts = scripted_judge()
        submission = parsed_submission(tmp_path, {"overview.pdf": []})

        result = self._grade(scenario_root, judge, submission, check_index=1)
        assert result.passed is False
        assert result.reasoning == "Not submitted: overview.tex"
        assert self._grade(scenario_root, judge, None).passed is False
        assert prompts == []

    def test_the_judge_verdict_is_recorded(self, scenario_root: Path, tmp_path: Path) -> None:
        judge, prompts = scripted_judge(verdict(True))
        submission = parsed_submission(tmp_path, {"overview.tex": [{"type": "text", "text": "x"}]})

        result = self._grade(scenario_root, judge, submission)
        assert result.passed is True
        assert result.reasoning == "because"
        assert prompts[0][1] == "Grade strictly."

    def test_unparseable_replies_leave_the_check_ungraded(
        self, scenario_root: Path, tmp_path: Path, no_waits
    ) -> None:
        judge, prompts = scripted_judge("?", "?", "?")
        submission = parsed_submission(tmp_path, {"overview.tex": []})

        result = self._grade(scenario_root, judge, submission)
        assert result.passed is None
        assert result.raw_reply == "?"
        assert "judge_parse_error" in (result.error or "")
        assert len(prompts) == 3

    def test_a_failing_judge_leaves_the_check_ungraded(
        self, scenario_root: Path, tmp_path: Path, no_waits
    ) -> None:
        judge, _ = scripted_judge(RuntimeError("rate limited"))
        submission = parsed_submission(tmp_path, {"overview.tex": []})

        result = self._grade(scenario_root, judge, submission)
        assert result.passed is None
        assert result.error == "judge_error: rate limited"


def trial(task_id: str, results: list[bool | None], attempt: int = 0, **fields: Any) -> TrialResult:
    verdicts = [
        CheckVerdict(check_id=f"{task_id}-{i}", check_type="A" if i % 2 else "C", passed=r)
        for i, r in enumerate(results)
    ]
    return TrialResult(task_id=task_id, attempt=attempt, verdicts=verdicts, **fields)


class TestMetrics:
    def test_pass_rates_pool_checks_and_leave_out_failed_trials(self) -> None:
        trials = [
            trial("w1_t1", [True, False, True, True], completion_reason="finished"),
            trial("w1_t2", [False, False], completion_reason="abandoned"),
            trial("w1_t3", [True], error="sandbox_error: boom"),
        ]
        metrics = compute_metrics(trials)

        assert next(iter(metrics)) == "rubric_pass_rate"
        assert metrics["rubric_pass_rate"] == pytest.approx(3 / 6)
        assert metrics["rubric_pass_rate_w1_t1"] == pytest.approx(3 / 4)
        assert "rubric_pass_rate_w1_t3" not in metrics
        assert metrics["critical_insight_pass_rate"] == pytest.approx(2 / 3)
        assert metrics["accuracy_pass_rate"] == pytest.approx(1 / 3)
        assert metrics["finish_rate"] == 0.5
        assert metrics["abandon_rate"] == 0.5
        assert metrics["num_failed_trials"] == 1.0
        assert metrics["num_graded_checks"] == 6.0
        assert "rubric_pass_rate_stderr" not in metrics

    def test_repeated_attempts_report_a_standard_error(self) -> None:
        metrics = compute_metrics(
            [trial("w1_t1", [True, True], attempt=0), trial("w1_t1", [False, False], attempt=1)]
        )
        assert metrics["rubric_pass_rate"] == 0.5
        assert metrics["rubric_pass_rate_stderr"] == pytest.approx(0.5)


class FakeSandboxManager:
    """Stands in for the agent and parse containers.

    The agent container answers the deliverable copies; the parse container writes
    a manifest to the host directory mounted at the parsed path.
    """

    instances: list[FakeSandboxManager] = []

    def __init__(self, configs, owner="default") -> None:
        self.config = configs[0]
        self.owner = owner
        self.commands: list[str] = []
        self.stopped = False
        FakeSandboxManager.instances.append(self)

    def mount(self, target: str) -> Path:
        for spec in self.config.docker_args:
            fields = dict(part.split("=", 1) for part in spec.split(",") if "=" in part)
            if fields.get("dst") == target:
                return Path(fields["src"])
        raise AssertionError(f"nothing mounted at {target}")

    async def start(self) -> None:
        pass

    def get_executor(self, capabilities):
        return self

    async def stop(self) -> None:
        self.stopped = True

    async def execute_command(self, command, timeout=None, capabilities=None, **_):
        self.commands.append(command)
        if command.startswith("python "):
            submitted = sorted(p.name for p in self.mount(bc_eval.SUBMISSION_DIR).iterdir())
            manifest = {"files": {n: [{"type": "text", "text": n}] for n in submitted}}
            (self.mount(bc_eval.PARSED_DIR) / "manifest.json").write_text(json.dumps(manifest))
        elif command.startswith("cp -- "):
            name = command.rsplit("/", 1)[-1].strip("'")
            (self.mount(bc_eval.OUTBOX_DIR) / name).write_text("deliverable")
        return ExecutionResult(success=True, output="", exit_code=0)


class FakeProvider:
    model_name = "fake-model"


class TestExecute:
    def setup_method(self) -> None:
        FakeSandboxManager.instances.clear()

    def _execute(
        self, scenario_root: Path, tmp_path: Path, monkeypatch, run_agent, judge, args=None
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        eval_obj = AABriefcaseLiteExternalEval()
        with (
            mock.patch("olmo_eval.harness.sandbox.SandboxManager", FakeSandboxManager),
            mock.patch.object(bc_eval, "download_scenario", return_value=scenario_root),
            mock.patch.object(bc_eval, "build_judge_fn", return_value=judge),
            mock.patch.object(eval_obj, "_sandbox_image", return_value="img"),
            mock.patch.object(eval_obj, "_run_agent", side_effect=run_agent),
        ):
            return asyncio.run(
                eval_obj.execute(FakeProvider(), args or {}, output_dir=str(tmp_path / "out"))
            )

    def test_finished_deliverables_are_parsed_and_graded(
        self, scenario_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        async def run_agent(scaffold, task, scenario, provider, bc_args, trial):
            trial.completion_reason = "finished" if task.task_id == "w1_t1" else "max_turns"
            if task.task_id == "w1_t1":
                trial.finish = {
                    "tool": "finish",
                    "reason": "ok",
                    "paths": ["/home/user/overview.tex"],
                }

        # A deliverable left by an earlier run into the same directory is not graded.
        stale = tmp_path / "out/aa_briefcase_lite/w1_t2-0/submission/model.xlsx"
        stale.parent.mkdir(parents=True)
        stale.write_text("old")
        judge, prompts = scripted_judge(verdict(True), verdict(False))
        result = self._execute(scenario_root, tmp_path, monkeypatch, run_agent, judge)

        assert result.success, result.error
        # Both w1_t1 checks see the submitted tex and are judged; w1_t2 submitted
        # nothing, so its check fails without a call.
        assert result.metrics["rubric_pass_rate_w1_t1"] == pytest.approx(1 / 2)
        assert result.metrics["rubric_pass_rate_w1_t2"] == 0.0
        assert result.metrics["finish_rate"] == 0.5
        assert result.metadata["pairwise_checks_skipped"] == 1
        assert result.metadata["judge_config"] == {"model": "scripted"}

        agent = next(m for m in FakeSandboxManager.instances if m.owner == "briefcase-w1_t1:0")
        assert agent.commands == ["cp -- /home/user/overview.tex /mnt/outbox/overview.tex"]
        mounts = agent.config.docker_args
        assert any(m.endswith("dst=/home/user/shared,readonly") for m in mounts)
        assert any(m.endswith("dst=/mnt/outbox") for m in mounts)
        assert all(m.stopped for m in FakeSandboxManager.instances)

        by_id = {p["native_id"]: p for p in result.predictions or []}
        assert by_id["w1_t1:0"]["submitted"] == ["overview.tex"]
        assert by_id["w1_t2:0"]["instance_metrics"] == {"rubric_pass_rate": {"external": 0.0}}
        artifacts = tmp_path / "out/aa_briefcase_lite/w1_t1-0/submission/overview.tex"
        assert artifacts.read_text() == "deliverable"
        assert len(prompts) == 2
        assert not stale.exists()

    def test_a_run_stuck_past_its_timeout_fails_without_a_score(
        self, scenario_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        async def run_agent(scaffold, task, scenario, provider, bc_args, trial):
            await asyncio.sleep(60)

        judge, prompts = scripted_judge()
        with mock.patch.object(bc_eval, "AGENT_TIMEOUT_GRACE", 0.0):
            result = self._execute(
                scenario_root,
                tmp_path,
                monkeypatch,
                run_agent,
                judge,
                {"agent_timeout": "0.1", "task_ids": "w1_t1"},
            )

        assert not result.success
        assert "agent_stuck" in (result.error or "")
        assert prompts == []

    def test_a_model_server_error_fails_the_run_without_a_score(
        self, scenario_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        async def run_agent(scaffold, task, scenario, provider, bc_args, trial):
            trial.completion_reason = "error"
            trial.error = "agent_error: server down"

        judge, prompts = scripted_judge()
        result = self._execute(scenario_root, tmp_path, monkeypatch, run_agent, judge)

        assert not result.success
        assert "2 of 2 trials failed" in (result.error or "")
        assert result.metrics["num_failed_trials"] == 2.0
        assert all(p["instance_metrics"] == {} for p in result.predictions or [])
        assert prompts == []

    def test_args_parse_from_strings(self) -> None:
        args = BriefcaseArgs.from_dict(
            {"task_ids": "w1_t1,w1_t3", "n_attempts": "3", "judge_reasoning_effort": "none"}
        )
        assert args.task_ids == ["w1_t1", "w1_t3"]
        assert args.n_attempts == 3
        assert args.judge_reasoning_effort is None
        assert args.sampling_params.fit_max_tokens_to_context
        assert args.agent_timeout is None

    def test_eval_is_registered(self) -> None:
        eval_obj = get_external_eval("aa_briefcase_lite")
        assert eval_obj.scaffold == "stirrup"
        assert eval_obj.required_secrets == ("OPENAI_API_KEY",)


class TestParser:
    """The parser's paths that need none of the sandbox's packages."""

    def test_text_and_unsupported_files(self, tmp_path: Path) -> None:
        submission = tmp_path / "submission"
        submission.mkdir()
        (submission / "briefing.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nHello\n")
        (submission / "notes.bin").write_bytes(b"\x00\x01")
        out = tmp_path / "parsed"

        subprocess.run([sys.executable, "-B", str(PARSER), str(submission), str(out)], check=True)

        files = json.loads((out / "manifest.json").read_text())["files"]
        assert files["briefing.srt"] == [
            {"type": "text", "text": "1\n00:00:00,000 --> 00:00:02,000\nHello\n"}
        ]
        assert files["notes.bin"][0]["type"] == "note"
