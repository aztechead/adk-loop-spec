"""The request graph: deterministic routing, and full offline assembly."""

from __future__ import annotations

from pathlib import Path

from google.adk.workflow import DEFAULT_ROUTE

from devteam import build_app
from devteam.config import AppConfig
from devteam.graph import CHANGE_ROUTE, QUESTION_ROUTE, route_request


def routes_of(label: str) -> list[str]:
    event = route_request(label)
    assert event.actions is not None
    return list(event.actions.route or [])


def test_feature_and_bug_share_the_change_route() -> None:
    assert routes_of("FEATURE") == [CHANGE_ROUTE]
    assert routes_of(" bug \n") == [CHANGE_ROUTE]


def test_question_routes_to_qa() -> None:
    assert routes_of("question") == [QUESTION_ROUTE]


def test_unknown_label_takes_the_default_route() -> None:
    assert routes_of("BANANA") == [DEFAULT_ROUTE]


def test_app_assembles_offline(config: AppConfig, tmp_path: Path) -> None:
    """The whole App — graph, loop-spec mount, plugins — builds with no network."""
    app = build_app(config, project_dir=tmp_path)
    assert app.root_agent.name == "devteam_workflow"
    assert [plugin.name for plugin in app.plugins] == ["loop_spec", "memory_commit"]
