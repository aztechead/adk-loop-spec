"""The manager loop: the stall rule, the hand-off prompt, and a full offline run."""

import json
from pathlib import Path

import pytest
from google.adk.agents import Context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import FunctionNode
from google.genai import types

from devteam.config import AppConfig
from devteam.cycle import Checklist, ChecklistItem, CycleResult, read_checklist
from devteam.manager import (
    AUTO_PROMPT,
    LEDGER_KEY,
    RESULT_KEY,
    RESUME_PROMPT,
    ROUND_KEY,
    Decision,
    HumanDecision,
    Ledger,
    LedgerEntry,
    PhaseVerdict,
    RoundReport,
    build_manager_loop,
    next_prompt,
    parse_verdict,
    render_progress,
)
from devteam.runtime import policy_oracle, run_turn, text_message
from tests.conftest import ScriptedLlm, base_raw

NOW = "2026-09-04T00:00:00+00:00"


def entries(*ticks: tuple[str, int]) -> Ledger:
    return Ledger(
        entries=[
            LedgerEntry(round=i + 1, phase=phase, ticked=ticked, total=5, at=NOW)
            for i, (phase, ticked) in enumerate(ticks)
        ]
    )


def test_stall_counts_rounds_without_a_new_tick_or_phase() -> None:
    assert entries().rounds_since_progress() == 0
    assert entries(("plan", 0)).rounds_since_progress() == 0
    assert entries(("plan", 0), ("execute", 0)).rounds_since_progress() == 0
    assert entries(("execute", 1), ("execute", 1), ("execute", 1)).rounds_since_progress() == 2
    assert entries(("execute", 1), ("execute", 1), ("execute", 2)).rounds_since_progress() == 0


def test_verdict_parses_from_any_shape_and_defaults_to_continue() -> None:
    assert parse_verdict({"decision": "HALT", "guidance": "stop"}).decision is Decision.HALT
    assert parse_verdict('{"decision": "MOVE_ON"}').decision is Decision.MOVE_ON
    assert parse_verdict("not json").decision is Decision.CONTINUE
    assert parse_verdict(None).decision is Decision.CONTINUE


def test_next_prompt_carries_resume_guidance_and_the_phase_prompt() -> None:
    config = AppConfig.model_validate(base_raw())
    prompt = next_prompt(
        config, PhaseVerdict(decision=Decision.CONTINUE, guidance="Keep tests green."), False
    )
    assert prompt.splitlines() == [
        RESUME_PROMPT,
        "Manager guidance: Keep tests green.",
        config.loop_spec.manager.phase_prompt,
    ]
    moving_on = next_prompt(config, PhaseVerdict(decision=Decision.MOVE_ON), stalled=False)
    assert "Do not polish" in moving_on
    assert "Do not polish" in next_prompt(config, PhaseVerdict(decision=Decision.CONTINUE), True)


def test_checklist_reads_plan_tasks_and_their_status(tmp_path: Path) -> None:
    assert read_checklist(tmp_path, None) == Checklist()
    assert read_checklist(tmp_path, "slug") == Checklist(slug="slug")
    feature = tmp_path / ".loop-spec" / "features" / "slug"
    feature.mkdir(parents=True)
    (feature / "tasks.json").write_text(
        json.dumps(
            [
                {"id": "task-001", "subject": "scaffold", "status": "done"},
                {"id": "task-002", "subject": "endpoint"},
                {"not": "a task"},
            ]
        )
    )
    checklist = read_checklist(tmp_path, "slug")
    assert checklist.items == [
        ChecklistItem(id="task-001", subject="scaffold", done=True),
        ChecklistItem(id="task-002", subject="endpoint", done=False),
    ]
    assert (checklist.ticked, checklist.total) == (1, 2)


def test_progress_page_renders_boxes_and_the_ledger() -> None:
    assert "No round has completed" in render_progress(None, None)
    report = RoundReport(
        round=2,
        result=CycleResult(status="paused", reason="phase-handoff", phase_reached="execute"),
        checklist=Checklist(slug="s", items=[ChecklistItem(id="t1", subject="a <b>", done=True)]),
        ledger=entries(("plan", 0), ("execute", 1)),
        stalled=False,
        exhausted=False,
    )
    page = render_progress(report, None)
    assert "1 of 1 boxes ticked" in page and "&#9745; t1 a &lt;b&gt;" in page
    assert "<polyline" in page and "<td>execute</td>" in page
    assert "outcome=delivered" in render_progress(report, CycleResult(outcome="delivered"))


class FakeLoopSpec:
    """A stand-in for the loop-spec working agent: writes what the real one would.

    Each call is one phase in ``supervised`` mode: the first two hand off after
    PLAN and EXECUTE, the third delivers. It records every prompt it was given.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.prompts: list[str] = []

    def write(self, name: str, payload: object) -> None:
        path = self.project_dir / ".loop-spec" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def loop_spec(self, ctx: Context, node_input: str) -> str:
        self.prompts.append(node_input)
        tasks = [
            {"id": "task-001", "subject": "scaffold"},
            {"id": "task-002", "subject": "endpoint"},
        ]
        match len(self.prompts):
            case 1:
                result = {"status": "paused", "reason": "phase-handoff", "phaseReached": "plan"}
            case 2:
                tasks = [t | {"status": "done"} for t in tasks]
                result = {"status": "paused", "reason": "phase-handoff", "phaseReached": "execute"}
            case _:
                tasks = [t | {"status": "done"} for t in tasks]
                result = {
                    "status": "completed",
                    "outcome": "delivered",
                    "phaseReached": "deliver",
                    "converged": True,
                    "prUrl": "https://example/pr/1",
                }
        self.write("features/feat/tasks.json", tasks)
        self.write("last-result.json", result | {"slug": "feat"})
        return "phase done"


async def test_the_loop_drives_phases_until_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig.model_validate(base_raw())
    manager = ScriptedLlm(
        model="scripted",
        script={
            "You manage": PhaseVerdict(
                decision=Decision.CONTINUE, guidance="Ship it."
            ).model_dump_json()
        },
    )
    monkeypatch.setattr(
        "devteam.manager.model_for_agent", lambda role, cfg: manager.as_agent_model()
    )
    fake = FakeLoopSpec(tmp_path)
    workflow = build_manager_loop(
        config, tmp_path, FunctionNode(func=fake.loop_spec, name="loop_spec")
    )
    runner = Runner(node=workflow, app_name="t", session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name="t", user_id="u")

    outputs: list[object] = []
    authors: list[str] = []
    async for event in run_turn(
        runner,
        user_id="u",
        session_id=session.id,
        message=text_message("add a healthcheck endpoint"),
        oracle=policy_oracle(config.loop_spec.supervisor.oracle),
    ):
        authors.append(event.author)
        if isinstance(event.output, dict) and "outcome" in event.output:
            outputs.append(event.output)

    assert fake.prompts[0].startswith(AUTO_PROMPT.format(task="add a healthcheck endpoint"))
    assert fake.prompts[1].startswith(RESUME_PROMPT) and "Ship it." in fake.prompts[1]
    assert len(fake.prompts) == 3 and len(manager.requests) == 2
    # observe's DONE route and finish both carry the terminal result
    assert [CycleResult.model_validate(o).succeeded for o in outputs] == [True, True], authors

    stored = await runner.session_service.get_session(
        app_name="t", user_id="u", session_id=session.id
    )
    assert stored is not None
    assert stored.state[ROUND_KEY] == 3
    ledger = Ledger.model_validate(stored.state[LEDGER_KEY])
    assert [(e.phase, e.ticked, e.total) for e in ledger.entries] == [
        ("plan", 0, 2),
        ("execute", 2, 2),
        ("deliver", 2, 2),
    ]
    assert CycleResult.model_validate(stored.state[RESULT_KEY]).pr_url == "https://example/pr/1"


async def test_a_halt_pauses_for_a_human_and_a_stop_ends_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HALT surfaces as a pending request_input; the typed human reply resumes the graph."""
    config = AppConfig.model_validate(base_raw())
    manager = ScriptedLlm(
        model="scripted",
        script={
            "You manage": PhaseVerdict(
                decision=Decision.HALT, guidance="Ask first."
            ).model_dump_json()
        },
    )
    monkeypatch.setattr(
        "devteam.manager.model_for_agent", lambda role, cfg: manager.as_agent_model()
    )
    fake = FakeLoopSpec(tmp_path)
    workflow = build_manager_loop(
        config, tmp_path, FunctionNode(func=fake.loop_spec, name="loop_spec")
    )
    runner = Runner(node=workflow, app_name="t", session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name="t", user_id="u")

    pending: list[types.FunctionCall] = []
    async for event in run_turn(
        runner,
        user_id="u",
        session_id=session.id,
        message=text_message("add a healthcheck endpoint"),
        oracle=policy_oracle(config.loop_spec.supervisor.oracle),
    ):
        if event.long_running_tool_ids and event.content:
            pending += [p.function_call for p in event.content.parts or [] if p.function_call]

    assert len(fake.prompts) == 1  # the loop stopped after the first hand-off
    (call,) = pending
    assert call.name == "adk_request_input" and call.args
    assert "the manager halted: Ask first." in str(call.args["message"])
    assert call.args["response_schema"]["title"] == HumanDecision.__name__
    stored = await runner.session_service.get_session(
        app_name="t", user_id="u", session_id=session.id
    )
    assert stored is not None and stored.state.get(RESULT_KEY) is None

    reply = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response=HumanDecision(resume=False).model_dump(),
                )
            )
        ],
    )
    async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=reply):
        pass
    stored = await runner.session_service.get_session(
        app_name="t", user_id="u", session_id=session.id
    )
    assert stored is not None
    result = CycleResult.model_validate(stored.state[RESULT_KEY])
    assert (result.outcome, result.succeeded) == ("halted", False)
    assert len(fake.prompts) == 1
