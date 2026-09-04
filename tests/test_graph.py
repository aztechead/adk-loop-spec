"""The request graph: deterministic routing, and a full offline run."""

import json
from pathlib import Path

import pytest
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import DEFAULT_ROUTE, FunctionNode

from devteam.agents import QA_STATE_KEY, Category, IntakeResult, QaAnswer
from devteam.app import MEMORY_WATERMARK_KEY, MemoryCommitPlugin, build_app
from devteam.config import AppConfig
from devteam.cycle import CycleResult
from devteam.graph import (
    CHANGE_ROUTE,
    QUESTION_ROUTE,
    Clarification,
    build_graph,
    decide,
    peer_route,
)
from devteam.manager import AUTO_PROMPT, RESULT_KEY, Decision, PhaseVerdict, build_manager_loop
from devteam.runtime import policy_oracle, run_turn, text_message
from tests.conftest import ScriptedLlm, base_raw
from tests.test_manager import FakeLoopSpec

PEERS = frozenset({"platform_team"})


def routed(
    node_input: IntakeResult | dict[str, object] | str, stored: object = None
) -> tuple[list[bool | int | str], object]:
    event = decide(node_input, stored, PEERS)
    route = event.actions.route
    assert isinstance(route, list)
    return route, event.output


def test_feature_and_bug_share_the_change_route() -> None:
    assert routed(IntakeResult(category=Category.FEATURE, request="add x")) == (
        [CHANGE_ROUTE],
        "add x",
    )
    assert routed({"category": "BUG", "request": "fix y"}) == ([CHANGE_ROUTE], "fix y")


def test_question_routes_to_qa_with_the_request_text() -> None:
    verdict = json.dumps({"category": "QUESTION", "request": "how is it deployed?"})
    assert routed(verdict) == ([QUESTION_ROUTE], "how is it deployed?")


def test_known_peer_team_wins_over_the_category() -> None:
    verdict = IntakeResult(category=Category.BUG, request="fix it", team="platform_team")
    assert routed(verdict) == ([peer_route("platform_team")], "fix it")


def test_unknown_team_falls_back_to_the_category() -> None:
    verdict = IntakeResult(category=Category.BUG, request="fix it", team="nobody")
    assert routed(verdict) == ([CHANGE_ROUTE], "fix it")


def test_unparseable_verdict_takes_the_default_route() -> None:
    assert routed("BANANA") == ([DEFAULT_ROUTE], "BANANA")


def test_the_stored_verdict_rescues_an_unusable_input() -> None:
    """Intake's output_key put the verdict in state; the router reads it back."""
    stored = {"category": "QUESTION", "request": "what is the SLA?", "team": None}
    assert routed("garbled", stored) == ([QUESTION_ROUTE], "what is the SLA?")


def test_a_human_clarification_is_a_verdict() -> None:
    reply = Clarification(category=Category.FEATURE, request="add dark mode")
    assert routed(reply.model_dump()) == ([CHANGE_ROUTE], "add dark mode")


def test_app_assembles_offline(config: AppConfig, tmp_path: Path) -> None:
    """The whole App — graph, manager loop, loop-spec mount, plugins — builds with no network."""
    app = build_app(config, project_dir=tmp_path)
    assert app.root_agent is not None and app.root_agent.name == "devteam_workflow"
    assert [plugin.name for plugin in app.plugins] == [
        "loop_spec",
        "reflect_retry_tool_plugin",
        "memory_commit",
    ]
    assert app.resumability_config is not None and app.resumability_config.is_resumable


async def test_question_reaches_qa_with_the_users_words(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the real Workflow, with a scripted model in place of the real one.

    Guards two contracts: the qa agent's last user turn must be the request
    itself, never the classifier's label; and its reply is a typed QaAnswer
    that also lands in session state.
    """
    request = "How do we deploy this service?"
    answer = QaAnswer(answer="Through the release pipeline.", sources=["memory: deploy notes"])
    llm = ScriptedLlm(
        model="scripted",
        script={
            "Classify": json.dumps({"category": "QUESTION", "request": request}),
            "Answer the user": answer.model_dump_json(),
        },
    )
    monkeypatch.setattr("devteam.agents.model_for_agent", lambda role, cfg: llm.as_agent_model())
    stub_change = LlmAgent(name="loop_spec", model=llm, instruction="never reached")
    runner = Runner(
        node=build_graph(config, stub_change),
        app_name="test",
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(app_name="test", user_id="u")

    texts: list[str] = []
    async for event in run_turn(
        runner,
        user_id="u",
        session_id=session.id,
        message=text_message(request),
        oracle=policy_oracle(config.loop_spec.supervisor.oracle),
    ):
        if event.author == "qa" and event.content:
            texts += [p.text for p in event.content.parts or [] if p.text]

    assert [QaAnswer.model_validate_json(t) for t in texts] == [answer]
    qa_request = llm.requests[-1]
    last_user = next(c for c in reversed(qa_request.contents) if c.role == "user")
    assert [p.text for p in last_user.parts or []] == [request]
    stored = await runner.session_service.get_session(
        app_name="test", user_id="u", session_id=session.id
    )
    assert stored is not None
    assert QaAnswer.model_validate(stored.state[QA_STATE_KEY]) == answer


async def test_a_feature_reaches_the_manager_loop_with_the_users_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CHANGE route: intake labels a feature, the manager loop drives loop-spec with it."""
    config = AppConfig.model_validate(base_raw())
    request = "add a healthcheck endpoint"
    llm = ScriptedLlm(
        model="scripted",
        script={
            "Classify": json.dumps({"category": "FEATURE", "request": request}),
            "You manage": PhaseVerdict(
                decision=Decision.CONTINUE, guidance="Go."
            ).model_dump_json(),
        },
    )
    monkeypatch.setattr("devteam.agents.model_for_agent", lambda role, cfg: llm.as_agent_model())
    monkeypatch.setattr("devteam.manager.model_for_agent", lambda role, cfg: llm.as_agent_model())
    fake = FakeLoopSpec(tmp_path)
    loop = build_manager_loop(config, tmp_path, FunctionNode(func=fake.loop_spec, name="loop_spec"))
    runner = Runner(
        node=build_graph(config, loop), app_name="test", session_service=InMemorySessionService()
    )
    session = await runner.session_service.create_session(app_name="test", user_id="u")
    async for _ in run_turn(
        runner,
        user_id="u",
        session_id=session.id,
        message=text_message(request),
        oracle=policy_oracle(config.loop_spec.supervisor.oracle),
    ):
        pass

    assert fake.prompts[0].startswith(AUTO_PROMPT.format(task=request))
    assert len(fake.prompts) == 3
    stored = await runner.session_service.get_session(
        app_name="test", user_id="u", session_id=session.id
    )
    assert stored is not None
    assert CycleResult.model_validate(stored.state[RESULT_KEY]).succeeded


async def test_memory_commits_only_new_events_across_turns(config: AppConfig) -> None:
    """The watermark lives in session state, so a second turn commits only its own events."""
    llm = ScriptedLlm(model="scripted", script={"Reply": "noted"})
    agent = LlmAgent(name="echo", model=llm, instruction="Reply briefly.")
    memory = InMemoryMemoryService()
    runner = Runner(
        app=App(name="test", root_agent=agent, plugins=[MemoryCommitPlugin()]),
        session_service=InMemorySessionService(),
        memory_service=memory,
    )
    session = await runner.session_service.create_session(app_name="test", user_id="u")

    async def turn(text: str) -> int:
        async for _ in runner.run_async(
            user_id="u", session_id=session.id, new_message=text_message(text)
        ):
            pass
        stored = await runner.session_service.get_session(
            app_name="test", user_id="u", session_id=session.id
        )
        assert stored is not None
        return int(stored.state[MEMORY_WATERMARK_KEY])

    first = await turn("one")
    second = await turn("two")
    assert 0 < first < second
    committed = memory._session_events[("test", "u")][session.id]
    assert [p.text for e in committed for p in ((e.content.parts or []) if e.content else [])] == [
        "one",
        "noted",
        "two",
        "noted",
    ]
