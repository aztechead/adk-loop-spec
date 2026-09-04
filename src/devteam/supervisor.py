"""Drive an unattended loop-spec run end to end — the supervisor side.

loop-spec's supervisor interface names four ports (its
``docs/loop-spec/supervisor-interface.md``); this module lands each on its
native ADK seam, with policy coming from ``loop_spec.supervisor`` and
``loop_spec.manager`` in YAML:

    decision oracle  runtime.policy_oracle answers pending ``get_user_choice`` calls
    lifecycle        the manager loop (devteam.manager) re-issues the cycle after each
                     ``phase-handoff`` in a fresh context; an interrupted invocation is
                     resumed by id; a dead run is reconciled through cycle-reconcile.sh
    state store      profile env names loop-spec's store-mirror adapter + a directory
    event sink       profile env names loop-spec's append-sink adapter + a file

The run succeeded only when the terminal result says ``outcome`` is
``delivered`` or ``no-change-needed`` AND ``converged`` is true — ``status``
alone is not success, per loop-spec's agent output contract.
"""

import json
from pathlib import Path

from google.adk.runners import Runner

from devteam.app import build_loop_spec_app, runner_for
from devteam.config import AppConfig
from devteam.cycle import CycleResult, read_last_result, run_script
from devteam.manager import RESULT_KEY, ROUND_KEY
from devteam.runtime import policy_oracle, run_turn, text_message

USER_ID = "supervisor"

# Adapters bundled in the loop-spec checkout, relative to its root.
STORE_MIRROR = Path("lib/supervisor/store-mirror.sh")
APPEND_SINK = Path("examples/supervisor/append-sink.sh")
PROFILE = Path("lib/profile.sh")


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
    check = run_script(config.loop_spec.root.resolve(), PROFILE, "validate", cwd=project_dir)
    if check.returncode != 0:
        raise RuntimeError(f"loop-spec rejected {profile_path}: {check.stderr.strip()}")
    return profile_path


class Supervisor:
    """Owns one supervised run of the manager loop against one project checkout."""

    def __init__(self, config: AppConfig, project_dir: Path) -> None:
        self._config = config
        self._project_dir = project_dir
        self._oracle = policy_oracle(config.loop_spec.supervisor.oracle)

    def _build_runner(self) -> Runner:
        return runner_for(self._config, build_loop_spec_app(self._config, self._project_dir))

    async def run(self, task: str) -> CycleResult:
        """One invocation of the manager loop, answering questions as they arrive.

        If the invocation dies mid-turn and the App is resumable, it is resumed
        once by id before the failure is allowed to surface. The verdict comes
        from the loop's own state when it finished, else from loop-spec's
        result file.
        """
        ensure_profile(self._config, self._project_dir)
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
                message=text_message(task),
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
        finished = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=USER_ID, session_id=session.id
        )
        state = finished.state if finished else {}
        rounds = int(state.get(ROUND_KEY) or 0)
        if recorded := state.get(RESULT_KEY):
            return CycleResult.model_validate(recorded).model_copy(
                update={"handoffs": max(rounds - 1, 0)}
            )
        result = read_last_result(self._config.loop_spec.root, self._project_dir)
        return result.model_copy(update={"handoffs": max(rounds - 1, 0)})
