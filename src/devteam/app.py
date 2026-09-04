"""Assemble the ADK Apps and Runners: graph, plugins, and services, wired once.

This is the composition root — every other module builds one part and this
one connects them. Nothing here decides policy; it all comes from config.
"""

from pathlib import Path

from google.adk import Workflow
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event, EventActions
from google.adk.plugins import BasePlugin, ReflectAndRetryToolPlugin
from google.adk.runners import Runner

from devteam.config import AppConfig
from devteam.graph import build_graph
from devteam.loopspec import build_working_agent
from devteam.manager import build_manager_loop
from devteam.services import build_services

MEMORY_WATERMARK_KEY = "memory_committed_events"


class MemoryCommitPlugin(BasePlugin):
    """Feed each finished turn's new events into long-term memory.

    Only events since the last commit are sent, so Memory Bank extracts from
    fresh material instead of re-processing the whole session every turn. The
    watermark lives in session state, written through the session service as
    a state-delta event, so a run resumed in another process continues from
    the same mark instead of re-committing the whole session.
    """

    def __init__(self) -> None:
        super().__init__(name="memory_commit")

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        memory = invocation_context.memory_service
        session = invocation_context.session
        if memory is None:
            return
        start = int(session.state.get(MEMORY_WATERMARK_KEY) or 0)
        fresh = [e for e in session.events[start:] if e.content and e.content.parts]
        if fresh:
            await memory.add_events_to_memory(
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
                events=fresh,
            )
        await invocation_context.session_service.append_event(
            session,
            Event(
                invocation_id=invocation_context.invocation_id,
                author=self.name,
                actions=EventActions(state_delta={MEMORY_WATERMARK_KEY: len(session.events)}),
            ),
        )


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


def _manager_loop(config: AppConfig, project_dir: Path) -> tuple[Workflow, BasePlugin]:
    """The manager loop over the mounted loop-spec agent, plus that agent's plugin.

    The loop-spec agent and its plugin share one bridge, so both must land on
    the same App together.
    """
    agent, plugin = build_working_agent(config, project_dir)
    return build_manager_loop(config, project_dir, agent), plugin


def build_app(config: AppConfig, project_dir: Path | None = None) -> App:
    """The dev-team App: the request graph, with the manager loop shipping changes.

    ``project_dir`` is the repository loop-spec works on (defaults to the
    current directory).
    """
    loop, plugin = _manager_loop(config, project_dir or Path.cwd())
    return _app(config, config.app.name, build_graph(config, loop), [plugin])


def build_loop_spec_app(config: AppConfig, project_dir: Path) -> App:
    """The manager loop as its own App, for supervised runs."""
    loop, plugin = _manager_loop(config, project_dir)
    return _app(config, "loop_spec", loop, [plugin])


def build_runner(config: AppConfig, project_dir: Path | None = None) -> Runner:
    """A Runner over the dev-team App with the configured session and memory services."""
    return runner_for(config, build_app(config, project_dir))


def runner_for(config: AppConfig, app: App) -> Runner:
    services = build_services(config)
    return Runner(app=app, session_service=services.session, memory_service=services.memory)
