"""Turn a PLAN task list into waves: what can run at once, and what must wait.

loop-spec's EXECUTE phase orders tasks by two kinds of edge, and this module
reproduces both so a wave never holds two tasks that would collide:

- an explicit ``blockedBy`` edge, declared by the planner;
- a synthetic edge between any two tasks whose ``files`` overlap (lower id
  first), which is what loop-spec's step 2b adds from its conflict table.

A wave is one topological layer, chunked to the implementer cap. Waves run in
sequence; the tasks inside a wave run at once, each in its own git worktree.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from devteam.cycle import LOOP_SPEC_DIR


class PlanTask(BaseModel):
    """One PLAN task, with the fields the implementer brief and the DAG need."""

    id: str
    subject: str = ""
    files: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list, alias="blockedBy")
    acceptance_criteria: list[str] = Field(default_factory=list, alias="acceptanceCriteria")
    verify_command: str = Field(default="true", alias="verifyCommand")
    read_first: list[str] = Field(default_factory=list, alias="readFirst")
    status: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def done(self) -> bool:
        return self.status == "done"


class WavePlan(BaseModel):
    """The ordered waves still to run, and which wave is next."""

    slug: str
    branch: str = Field(description="The feature branch every task branch merges into.")
    waves: list[list[str]] = Field(default_factory=list, description="Task ids per wave.")
    next_wave: int = 0

    @property
    def finished(self) -> bool:
        return self.next_wave >= len(self.waves)

    @property
    def current(self) -> list[str]:
        return self.waves[self.next_wave] if not self.finished else []


class DependencyCycleError(ValueError):
    """The task graph has a cycle, so no wave order exists."""


def tasks_path(project_dir: Path, slug: str) -> Path:
    return project_dir / LOOP_SPEC_DIR / "features" / slug / "tasks.json"


def read_tasks(project_dir: Path, slug: str) -> list[PlanTask]:
    """The feature's tasks.json as typed tasks; empty when PLAN has not written it."""
    path = tasks_path(project_dir, slug)
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        PlanTask.model_validate(task) for task in raw if isinstance(task, dict) and task.get("id")
    ]


def dependency_edges(tasks: Iterable[PlanTask]) -> dict[str, set[str]]:
    """Blockers per task: declared ``blockedBy`` plus one edge per file-overlapping pair."""
    ordered = sorted(tasks, key=lambda task: task.id)
    known = {task.id for task in ordered}
    blockers: dict[str, set[str]] = {
        task.id: {dep for dep in task.blocked_by if dep in known} for task in ordered
    }
    for index, earlier in enumerate(ordered):
        for later in ordered[index + 1 :]:
            if set(earlier.files) & set(later.files):
                blockers[later.id].add(earlier.id)
    return blockers


def plan_waves(tasks: Iterable[PlanTask], max_parallel: int) -> list[list[str]]:
    """Topological layers of the open tasks, each chunked to ``max_parallel``.

    Done tasks count as satisfied blockers and never appear in a wave.
    """
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    tasks = list(tasks)
    blockers = dependency_edges(tasks)
    done = {task.id for task in tasks if task.done}
    remaining = {task.id: blockers[task.id] - done for task in tasks if not task.done}
    waves: list[list[str]] = []
    while remaining:
        ready = sorted(task_id for task_id, deps in remaining.items() if not deps)
        if not ready:
            raise DependencyCycleError(f"tasks block each other: {sorted(remaining)}")
        for start in range(0, len(ready), max_parallel):
            waves.append(ready[start : start + max_parallel])
        for task_id in ready:
            del remaining[task_id]
        for deps in remaining.values():
            deps.difference_update(ready)
    return waves
