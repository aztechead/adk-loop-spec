"""Assemble the ADK App and Runner: graph, plugins, and services, wired once.

``build_app`` is the composition root — every other module builds one part and
this one connects them. Nothing here decides policy; it all comes from config.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.plugins import BasePlugin
from google.adk.runners import Runner

from .config import AppConfig
from .graph import build_graph
from .loopspec import build_working_agent
from .services import build_services


class MemoryCommitPlugin(BasePlugin):
    """Feed every finished conversation into long-term memory.

    With the in-memory backend this makes past chats searchable for the run's
    lifetime; on Agent Platform, Memory Bank extracts and consolidates durable
    memories from the same call.
    """

    def __init__(self) -> None:
        super().__init__(name="memory_commit")

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        memory = invocation_context.memory_service
        if memory is not None:
            await memory.add_session_to_memory(invocation_context.session)


def build_app(config: AppConfig, project_dir: Path | None = None) -> App:
    """The dev-team App: the request graph plus the loop-spec mount.

    ``project_dir`` is the repository loop-spec works on (defaults to the
    current directory). The loop-spec agent and its plugin share one bridge,
    so both land on this App together.
    """
    loop_spec_agent, loop_spec_plugin = build_working_agent(config, project_dir or Path.cwd())
    return App(
        name=config.app.name,
        root_agent=build_graph(config, loop_spec_agent),
        plugins=[loop_spec_plugin, MemoryCommitPlugin()],
    )


def build_runner(config: AppConfig, project_dir: Path | None = None) -> Runner:
    """A Runner over the App with the configured session and memory services."""
    services = build_services(config)
    return Runner(
        app=build_app(config, project_dir),
        session_service=services.session,
        memory_service=services.memory,
    )
