"""Drive an unattended loop-spec run end to end — the supervisor side.

loop-spec's supervisor interface names four ports (its
``docs/loop-spec/supervisor-interface.md``); this module implements each on
its native ADK seam:

    decision oracle  answer_question: resumes pending ``get_user_choice`` calls
    lifecycle        run(): re-invokes the cycle after each ``phase-handoff``
    state store      profile.json env (the checkout stays the working copy)
    event sink       profile.json env: every cycle event line reaches a program

The run succeeded only when the terminal result says ``outcome`` is
``delivered`` or ``no-change-needed`` AND ``converged`` is true — ``status``
alone is not success, per loop-spec's agent output contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from google.adk.events import Event
from google.adk.runners import Runner
from google.genai import types

from .config import AppConfig
from .loopspec import export_model_routes, load_extension
from .models import litellm_id
from .services import build_services

RECOMMENDED_MARK = "(Recommended)"
SUCCESS_OUTCOMES = frozenset({"delivered", "no-change-needed"})
_MAX_HANDOFFS = 25  # ponytail: hard stop against a spinning cycle; raise if real runs need more

AUTO_PROMPT = "Load the loop-spec auto skill and run: {task}"
RESUME_PROMPT = "Load the loop-spec cycle skill and run: autonomous"


@dataclass(frozen=True, slots=True)
class CycleResult:
    """The terminal verdict of one supervised run."""

    outcome: str
    converged: bool
    phase_reached: str
    handoffs: int

    @property
    def succeeded(self) -> bool:
        return self.outcome in SUCCESS_OUTCOMES and self.converged


def answer_question(options: list[str]) -> str:
    """The decision oracle: take the plugin's own recommendation.

    loop-spec puts its preferred answer first and marks it ``(Recommended)``;
    a policy that wants to override or halt replaces this one function.
    """
    for option in options:
        if RECOMMENDED_MARK in option:
            return option
    return options[0]


def read_last_result(project_dir: Path) -> dict[str, object]:
    """Parse .loop-spec/last-result.json, the cycle's durable outcome record."""
    result_path = project_dir / ".loop-spec" / "last-result.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"{result_path} was not written; the run died before producing a result"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def ensure_profile(project_dir: Path, event_sink: Path | None = None) -> Path:
    """Declare the supervised policy once, if the project has none yet.

    The profile is project policy and user-owned; an existing file is left
    exactly as it is.
    """
    profile_path = project_dir / ".loop-spec" / "profile.json"
    if profile_path.exists():
        return profile_path
    profile: dict[str, object] = {"preset": "supervised", "env": {"LOOP_SPEC_PHASE_HANDOFF": "1"}}
    if event_sink is not None:
        profile["env"] = {**profile["env"], "LOOP_SPEC_EVENT_SINK": str(event_sink)}  # type: ignore[dict-item]
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile_path


class Supervisor:
    """Owns one supervised loop-spec run against one project checkout."""

    def __init__(self, config: AppConfig, project_dir: Path) -> None:
        self._config = config
        self._project_dir = project_dir

    def _build_runner(self) -> Runner:
        """A fresh Runner over the loop-spec App — one per phase context."""
        export_model_routes(self._config)
        extension = load_extension(self._config.loop_spec.root)
        app = extension.build_app(
            self._project_dir,
            model=litellm_id(self._config.models.providers[self._config.loop_spec.agent]),
        )
        services = build_services(self._config)
        return Runner(app=app, session_service=services.session, memory_service=services.memory)

    async def _run_turn(self, prompt: str) -> None:
        """One invocation in a fresh session, answering questions as they arrive.

        A pending ``get_user_choice`` surfaces as a long-running function call;
        the supervisor answers it with a function response in the same session
        and the run continues until the turn ends.
        """
        runner = self._build_runner()
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="supervisor"
        )
        message: types.Content | None = types.Content(role="user", parts=[types.Part(text=prompt)])
        while message is not None:
            pending: types.Content | None = None
            async for event in runner.run_async(
                user_id="supervisor", session_id=session.id, new_message=message
            ):
                response = self._maybe_answer(event)
                if response is not None:
                    pending = response
            message = pending

    @staticmethod
    def _maybe_answer(event: Event) -> types.Content | None:
        """The function response answering an event's pending question, if any."""
        if not event.long_running_tool_ids or event.content is None:
            return None
        for part in event.content.parts or []:
            call = part.function_call
            if (
                call is not None
                and call.name == "get_user_choice"
                and call.id in event.long_running_tool_ids
            ):
                options = list((call.args or {}).get("options", []))
                if not options:
                    raise ValueError(f"get_user_choice call {call.id} carried no options")
                return types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=call.id,
                                name=call.name,
                                response={"result": answer_question(options)},
                            )
                        )
                    ],
                )
        return None

    async def run(self, task: str, event_sink: Path | None = None) -> CycleResult:
        """The lifecycle port: auto-route the task, resume across phase handoffs."""
        ensure_profile(self._project_dir, event_sink)
        prompt = AUTO_PROMPT.format(task=task)
        for handoff in range(_MAX_HANDOFFS):
            await self._run_turn(prompt)
            result = read_last_result(self._project_dir)
            if not (result.get("status") == "paused" and result.get("reason") == "phase-handoff"):
                return CycleResult(
                    outcome=str(result.get("outcome", "unknown")),
                    converged=bool(result.get("converged", False)),
                    phase_reached=str(result.get("phaseReached", "unknown")),
                    handoffs=handoff,
                )
            prompt = RESUME_PROMPT
        raise RuntimeError(f"cycle did not reach a terminal result in {_MAX_HANDOFFS} handoffs")
