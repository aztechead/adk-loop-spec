"""The request graph: deterministic routing, and a full offline run."""

import json
from pathlib import Path

import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import DEFAULT_ROUTE

from devteam import build_app
from devteam.agents import Category, IntakeResult
from devteam.config import AppConfig
from devteam.graph import CHANGE_ROUTE, QUESTION_ROUTE, build_graph, make_router, peer_route
from devteam.runtime import policy_oracle, run_turn, text_message
from tests.conftest import ScriptedLlm


def routed(
    node_input: IntakeResult | dict[str, object] | str,
) -> tuple[list[bool | int | str], object]:
    event = make_router(frozenset({"platform_team"}))(node_input)
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


def test_app_assembles_offline(config: AppConfig, tmp_path: Path) -> None:
    """The whole App — graph, loop-spec mount, plugins — builds with no network."""
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
    """End to end through the real Workflow, with a scripted model in place of LiteLLM.

    Guards the routing contract: the qa agent's last user turn must be the
    request itself, never the classifier's label.
    """
    request = "How do we deploy this service?"
    llm = ScriptedLlm(
        model="scripted",
        script={
            "Classify": json.dumps({"category": "QUESTION", "request": request}),
            "Answer the user": "Through the release pipeline.",
        },
    )
    monkeypatch.setattr("devteam.agents.model_for_agent", lambda role, cfg: llm)
    stub_loop_spec = LlmAgent(name="loop_spec", model=llm, instruction="never reached")
    runner = Runner(
        node=build_graph(config, stub_loop_spec),
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

    assert texts == ["Through the release pipeline."]
    qa_request = llm.requests[-1]
    last_user = next(c for c in reversed(qa_request.contents) if c.role == "user")
    assert [p.text for p in last_user.parts or []] == [request]
