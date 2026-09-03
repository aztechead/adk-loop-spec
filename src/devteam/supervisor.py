"""Drive an unattended loop-spec run end to end — the supervisor side.

loop-spec's supervisor interface names four ports (its
``docs/loop-spec/supervisor-interface.md``); this module lands each on its
native ADK seam, with policy coming from ``loop_spec.supervisor`` in YAML:

    decision oracle  runtime.policy_oracle answers pending ``get_user_choice`` calls
    lifecycle        run(): re-invokes the cycle after each ``phase-handoff``,
                     resumes an interrupted invocation, reconciles a dead run
    state store      profile env names loop-spec's store-mirror adapter + a directory
    event sink       profile env names loop-spec's append-sink adapter + a file

The run succeeded only when the terminal result says ``outcome`` is
``delivered`` or ``no-change-needed`` AND ``converged`` is true — ``status``
alone is not success, per loop-spec's agent output contract.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from google.adk.runners import Runner

from .app import build_loop_spec_app, runner_for
from .config import AppConfig
from .runtime import policy_oracle, run_turn, text_message

SUCCESS_OUTCOMES = frozenset({"delivered", "no-change-needed"})
AUTO_PROMPT = "Load the loop-spec auto skill and run: {task}"
RESUME_PROMPT = "Load the loop-spec cycle skill and run: autonomous"
USER_ID = "supervisor"

# Adapters bundled in the loop-spec checkout, relative to its root.
STORE_MIRROR = Path("lib/supervisor/store-mirror.sh")
APPEND_SINK = Path("examples/supervisor/append-sink.sh")
PROFILE = Path("lib/profile.sh")
RECONCILE = Path("lib/cycle-reconcile.sh")


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


def _bash(root: Path, script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / script), *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def profile_env(config: AppConfig) -> dict[str, str]:
    """The store and sink ports as loop-spec profile variables."""
    root = config.loop_spec.root.resolve()
    policy = config.loop_spec.supervisor
    env: dict[str, str] = {}
    if policy.store_dir is not None:
        env["LOOP_SPEC_STORE"] = str(root / STORE_MIRROR)
        env["LOOP_SPEC_STORE_DIR"] = str(policy.store_dir.resolve())
    if policy.events_file is not None:
        env["LOOP_SPEC_EVENT_SINK"] = str(root / APPEND_SINK)
        env["LOOP_SPEC_EVENT_SINK_FILE"] = str(policy.events_file.resolve())
    return env


def ensure_profile(config: AppConfig, project_dir: Path) -> Path:
    """Declare the supervised policy once, then let loop-spec validate it.

    The ``supervised`` preset already sets autonomy, oracle=supervisor, and
    phase handoff; only the two transport ports are added. The profile is
    project policy and user-owned: an existing file is left exactly as it is.
    """
    profile_path = project_dir / ".loop-spec" / "profile.json"
    if not profile_path.exists():
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {"preset": "supervised", "env": profile_env(config)}
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    check = _bash(config.loop_spec.root.resolve(), PROFILE, "validate", cwd=project_dir)
    if check.returncode != 0:
        raise RuntimeError(f"loop-spec rejected {profile_path}: {check.stderr.strip()}")
    return profile_path


def read_last_result(config: AppConfig, project_dir: Path) -> dict[str, object]:
    """The cycle's durable outcome record, reconciling a run that died before writing it."""
    result_path = project_dir / ".loop-spec" / "last-result.json"
    if not result_path.is_file():
        reconcile = _bash(
            config.loop_spec.root.resolve(),
            RECONCILE,
            "--reason",
            "supervisor: agent turn ended without a terminal result",
            cwd=project_dir,
        )
        if not result_path.is_file():
            raise FileNotFoundError(
                f"{result_path} was not written and cycle-reconcile could not produce it: "
                f"{reconcile.stderr.strip() or reconcile.stdout.strip()}"
            )
    return json.loads(result_path.read_text(encoding="utf-8"))


class Supervisor:
    """Owns one supervised loop-spec run against one project checkout."""

    def __init__(self, config: AppConfig, project_dir: Path) -> None:
        self._config = config
        self._project_dir = project_dir
        self._oracle = policy_oracle(config.loop_spec.supervisor.oracle)

    def _build_runner(self) -> Runner:
        """A fresh Runner over the loop-spec App — one per phase context."""
        return runner_for(self._config, build_loop_spec_app(self._config, self._project_dir))

    async def _run_turn(self, prompt: str) -> None:
        """One invocation in a fresh session, answering questions as they arrive.

        If the invocation dies mid-turn and the App is resumable, it is resumed
        once by id before the failure is allowed to surface.
        """
        runner = self._build_runner()
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=USER_ID
        )
        invocation_id: str | None = None
        try:
            async for event in run_turn(
                runner,
                user_id=USER_ID,
                session_id=session.id,
                message=text_message(prompt),
                oracle=self._oracle,
            ):
                invocation_id = event.invocation_id or invocation_id
        except Exception:
            if invocation_id is None or not self._config.app.resumable:
                raise
            async for _ in runner.run_async(
                user_id=USER_ID, session_id=session.id, invocation_id=invocation_id
            ):
                pass

    async def run(self, task: str) -> CycleResult:
        """The lifecycle port: auto-route the task, resume across phase handoffs."""
        ensure_profile(self._config, self._project_dir)
        limit = self._config.loop_spec.supervisor.max_handoffs
        prompt = AUTO_PROMPT.format(task=task)
        for handoff in range(limit):
            await self._run_turn(prompt)
            result = read_last_result(self._config, self._project_dir)
            if not (result.get("status") == "paused" and result.get("reason") == "phase-handoff"):
                return CycleResult(
                    outcome=str(result.get("outcome", "unknown")),
                    converged=bool(result.get("converged", False)),
                    phase_reached=str(result.get("phaseReached", "unknown")),
                    handoffs=handoff,
                )
            prompt = RESUME_PROMPT
        raise RuntimeError(f"cycle did not reach a terminal result in {limit} handoffs")
