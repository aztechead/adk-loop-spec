"""The devteam command line: check, chat, serve, supervise.

Thin by design — each subcommand parses arguments, loads config, and calls one
function from the module that owns the behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from google.genai import types

from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .loopspec import model_routes
from .models import litellm_id


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="repository loop-spec works on (default: current directory)",
    )


def cmd_check(config: AppConfig, _: argparse.Namespace) -> int:
    """Validate the config and show every resolved model route. No network."""
    print(f"app: {config.app.name}")
    print(f"services: {config.services.backend}")
    for agent, provider_key in sorted(config.models.agents.items()):
        print(f"agent {agent}: {litellm_id(config.models.providers[provider_key])}")
    print(f"loop-spec agent: {litellm_id(config.models.providers[config.loop_spec.agent])}")
    for name, value in sorted(model_routes(config).items()):
        print(f"{name}={value}")
    for peer in config.a2a.peers:
        print(f"peer {peer.name}: {peer.agent_card_url}")
    print("config: ok")
    return 0


def cmd_chat(config: AppConfig, args: argparse.Namespace) -> int:
    """Send one message through the request graph and print the responses."""
    from .app import build_runner  # deferred: builds the loop-spec mount

    async def run() -> None:
        runner = build_runner(config, args.project_dir)
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=args.user
        )
        message = types.Content(role="user", parts=[types.Part(text=args.message)])
        async for event in runner.run_async(
            user_id=args.user, session_id=session.id, new_message=message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"[{event.author}] {part.text}")

    asyncio.run(run())
    return 0


def cmd_serve(config: AppConfig, args: argparse.Namespace) -> int:
    """Expose this instance to its peers over A2A."""
    from .a2a import serve  # deferred: builds the loop-spec mount

    serve(config, args.project_dir)
    return 0


def cmd_supervise(config: AppConfig, args: argparse.Namespace) -> int:
    """Run one loop-spec task unattended and report the terminal verdict."""
    from .supervisor import Supervisor  # deferred: builds the loop-spec mount

    project_dir = (args.project_dir or Path.cwd()).resolve()
    result = asyncio.run(Supervisor(config, project_dir).run(args.task))
    print(
        f"outcome={result.outcome} converged={result.converged} "
        f"phase={result.phase_reached} handoffs={result.handoffs}"
    )
    return 0 if result.succeeded else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devteam", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate config offline")
    check.set_defaults(handler=cmd_check)

    chat = commands.add_parser("chat", help="send one message through the graph")
    chat.add_argument("message")
    chat.add_argument("--user", default="local")
    chat.set_defaults(handler=cmd_chat)

    serve = commands.add_parser("serve", help="serve the A2A endpoint")
    serve.set_defaults(handler=cmd_serve)

    supervise = commands.add_parser("supervise", help="run a loop-spec task unattended")
    supervise.add_argument("task")
    supervise.set_defaults(handler=cmd_supervise)

    for sub in (check, chat, serve, supervise):
        _add_common(sub)

    args = parser.parse_args(argv)
    return args.handler(load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
