"""Assemble the ADK Apps and Runners: graph, plugins, and services, wired once.

This is the composition root — every other module builds one part and this
one connects them. Nothing here decides policy; it all comes from config.
"""

from pathlib import Path

from google.adk import Workflow
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App, ResumabilityConfig
from google.adk.plugins import BasePlugin, ReflectAndRetryToolPlugin
from google.adk.runners import Runner

from .config import AppConfig
from .graph import build_graph
from .loopspec import build_working_agent
from .services import build_services


class MemoryCommitPlugin(BasePlugin):
    """Feed each finished turn's new events into long-term memory.

    Only events since the last commit are sent, so Memory Bank extracts from
    fresh material instead of re-processing the whole session every turn.
    """

    def __init__(self) -> None:
        super().__init__(name="memory_commit")
        # ponytail: per-process watermark; move into session state if runs span processes
        self._committed: dict[str, int] = {}

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        memory = invocation_context.memory_service
        session = invocation_context.session
        if memory is None:
            return
        start = self._committed.get(session.id, 0)
        fresh = [e for e in session.events[start:] if e.content and e.content.parts]
        if fresh:
            await memory.add_events_to_memory(
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
                events=fresh,
            )
        self._committed[session.id] = len(session.events)


def _app(
    config: AppConfig, name: str, root: BaseAgent | Workflow, plugins: list[BasePlugin]
) -> App:
    """An App with the shared plugins and lifecycle settings every root gets."""
    return App(
        name=name,
        root_agent=root,
        plugins=[
            *plugins,
            ReflectAndRetryToolPlugin(max_retries=config.app.tool_retries),
            MemoryCommitPlugin(),
        ],
        resumability_config=ResumabilityConfig(is_resumable=config.app.resumable),
    )


def build_app(config: AppConfig, project_dir: Path | None = None) -> App:
    """The dev-team App: the request graph plus the loop-spec mount.

    ``project_dir`` is the repository loop-spec works on (defaults to the
    current directory). The loop-spec agent and its plugin share one bridge,
    so both land on this App together.
    """
    agent, plugin = build_working_agent(config, project_dir or Path.cwd())
    return _app(config, config.app.name, build_graph(config, agent), [plugin])


def build_loop_spec_app(config: AppConfig, project_dir: Path) -> App:
    """loop-spec's working agent as its own App, for supervised runs."""
    agent, plugin = build_working_agent(config, project_dir)
    return _app(config, "loop_spec", agent, [plugin])


def build_runner(config: AppConfig, project_dir: Path | None = None) -> Runner:
    """A Runner over the dev-team App with the configured session and memory services."""
    return runner_for(config, build_app(config, project_dir))


def runner_for(config: AppConfig, app: App) -> Runner:
    services = build_services(config)
    return Runner(app=app, session_service=services.session, memory_service=services.memory)
