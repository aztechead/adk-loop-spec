"""loop-spec's durable cycle artifacts, read into typed records.

Two files carry everything the supervisor and the manager loop need to know
about a run, both written by loop-spec itself under the project's
``.loop-spec`` directory:

    last-result.json                 the agent output contract (docs/loop-spec/agent-output-contract.md)
    features/<slug>/tasks.json       the PLAN task list, each task carrying ``status: done`` once published

Success is judged by ``outcome`` + ``converged``, never ``status`` alone, and a
run that died before writing its result is reconciled with loop-spec's own
``cycle-reconcile.sh`` so the record is always loop-spec's, not ours.
"""

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

SUCCESS_OUTCOMES = frozenset({"delivered", "no-change-needed"})
HANDOFF_STATUS = "paused"
HANDOFF_REASON = "phase-handoff"

RECONCILE = Path("lib/cycle-reconcile.sh")  # relative to the loop-spec checkout
LOOP_SPEC_DIR = Path(".loop-spec")
LAST_RESULT = LOOP_SPEC_DIR / "last-result.json"


class CycleResult(BaseModel):
    """One loop-spec terminal (or hand-off) result, as the contract spells it.

    Every field has a default because the contract is additive: consumers read
    each field with a default so older records stay valid.
    """

    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="ignore", frozen=True
    )

    status: str = "unknown"
    outcome: str = "unknown"
    reason: str | None = None
    summary: str = ""
    slug: str | None = None
    branch: str | None = None
    phase_reached: str | None = None
    converged: bool = False
    work_delivered: bool = False
    pr_url: str | None = None
    handoffs: int = Field(default=0, description="Phase hand-offs the supervisor drove.")

    @property
    def succeeded(self) -> bool:
        return self.outcome in SUCCESS_OUTCOMES and self.converged

    @property
    def is_handoff(self) -> bool:
        """The cycle paused after a durable phase and expects to be re-issued."""
        return self.status == HANDOFF_STATUS and self.reason == HANDOFF_REASON

    def handed_off_after(self, phase: str) -> bool:
        return self.is_handoff and (self.phase_reached or "").lower() == phase.lower()

    @property
    def feature_branch(self) -> str | None:
        """The branch loop-spec publishes tasks onto: the record's, else loop-spec's default."""
        if self.branch:
            return self.branch
        return f"feat/{self.slug}" if self.slug else None


class ChecklistItem(BaseModel):
    """One PLAN task, as the manager sees it."""

    id: str
    subject: str
    done: bool = False


class Checklist(BaseModel):
    """The PLAN task list of the active feature, with what has been published so far."""

    slug: str | None = None
    items: list[ChecklistItem] = Field(default_factory=list)

    @property
    def ticked(self) -> int:
        return sum(item.done for item in self.items)

    @property
    def total(self) -> int:
        return len(self.items)


def run_script(root: Path, script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one of loop-spec's bash scripts from the checkout at ``root``."""
    return subprocess.run(
        ["bash", str(root / script), *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def read_last_result(loop_spec_root: Path, project_dir: Path) -> CycleResult:
    """The cycle's durable outcome record, reconciling a run that died before writing it."""
    result_path = project_dir / LAST_RESULT
    if not result_path.is_file():
        reconcile = run_script(
            loop_spec_root.resolve(),
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
    return CycleResult.model_validate_json(result_path.read_text(encoding="utf-8"))


def read_checklist(project_dir: Path, slug: str | None) -> Checklist:
    """The feature's tasks.json as a checklist; empty before PLAN has written it."""
    if not slug:
        return Checklist()
    tasks_path = project_dir / LOOP_SPEC_DIR / "features" / slug / "tasks.json"
    if not tasks_path.is_file():
        return Checklist(slug=slug)
    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    items = [
        ChecklistItem(
            id=str(task["id"]),
            subject=str(task.get("subject", "")),
            done=task.get("status") == "done",
        )
        for task in raw
        if isinstance(task, dict) and task.get("id")
    ]
    return Checklist(slug=slug, items=items)
