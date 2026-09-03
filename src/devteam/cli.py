"""The devteam command line: check, chat, serve, supervise.

Thin by design — each subcommand parses arguments, loads config, and calls one
function from the module that owns the behavior.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .loopspec import AGENT_DIR_VAR, environment
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
        default=Path.cwd(),
        help="repository loop-spec works on (default: current directory)",
    )


def cmd_check(config: AppConfig, args: argparse.Namespace) -> int:
    """Validate the config and show every resolved route. No network."""
    print(f"app: {config.app.name}")
    print(f"services: {config.services.backend}")
    for agent, provider_key in sorted(config.models.agents.items()):
        print(f"agent {agent}: {litellm_id(config.models.providers[provider_key])}")
    print(f"loop-spec agent: {litellm_id(config.models.providers[config.loop_spec.agent])}")
    env = environment(config, args.project_dir)
    for name, value in sorted(env.items()):
        print(f"{name}={value}")
    if AGENT_DIR_VAR not in env:
        print(
            f"warning: no CLI mount under {config.loop_spec.mount}; loop-spec's fleet rung "
            "needs `bash scripts/mount-loop-spec.sh`"
        )
    for peer in config.a2a.peers:
        print(f"peer {peer.name}: {peer.agent_card_url}")
    print("config: ok")
    return 0


def cmd_chat(config: AppConfig, args: argparse.Namespace) -> int:
    """Send one message through the request graph and print the responses."""
    from .app import build_runner  # deferred: builds the loop-spec mount
    from .runtime import policy_oracle, run_turn, text_message

    async def run() -> None:
        runner = build_runner(config, args.project_dir)
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=args.user
        )
        async for event in run_turn(
            runner,
            user_id=args.user,
            session_id=session.id,
            message=text_message(args.message),
            oracle=policy_oracle(config.loop_spec.supervisor.oracle),
        ):
            for part in (event.content.parts if event.content else None) or []:
                if part.text:
                    print(f"[{event.author}] {part.text}")

    asyncio.run(run())
    return 0


def cmd_serve(config: AppConfig, args: argparse.Namespace) -> int:
    """Expose this instance's Q&A agent to its peers over A2A."""
    from .a2a import serve  # deferred: builds services

    serve(config, args.project_dir)
    return 0


def cmd_supervise(config: AppConfig, args: argparse.Namespace) -> int:
    """Run one loop-spec task unattended and report the terminal verdict."""
    from .supervisor import Supervisor  # deferred: builds the loop-spec mount

    result = asyncio.run(Supervisor(config, args.project_dir.resolve()).run(args.task))
    print(
        f"outcome={result.outcome} converged={result.converged} "
        f"phase={result.phase_reached} handoffs={result.handoffs}"
    )
    return 0 if result.succeeded else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # secrets from ./.env, never overriding what the shell already set
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
