"""Wave planning: declared and file-overlap edges, chunking, done tasks, cycles."""

import json
from pathlib import Path

import pytest

from devteam.waves import DependencyCycleError, PlanTask, dependency_edges, plan_waves, read_tasks


def task(
    id: str, files: list[str] | None = None, blocked_by: list[str] | None = None, **extra: object
) -> PlanTask:
    return PlanTask.model_validate(
        {"id": id, "files": files or [], "blockedBy": blocked_by or [], **extra}
    )


def test_file_overlap_adds_an_edge_from_the_lower_id() -> None:
    edges = dependency_edges(
        [task("t2", ["a.py"]), task("t1", ["a.py", "b.py"]), task("t3", ["c.py"])]
    )
    assert edges == {"t1": set(), "t2": {"t1"}, "t3": set()}


def test_waves_follow_dependencies_and_the_cap() -> None:
    tasks = [
        task("t1", ["a.py"]),
        task("t2", ["b.py"]),
        task("t3", ["c.py"], blocked_by=["t1"]),
        task("t4", ["d.py"]),
        task("t5", ["e.py"], blocked_by=["t3", "t2"]),
    ]
    assert plan_waves(tasks, max_parallel=2) == [["t1", "t2"], ["t4"], ["t3"], ["t5"]]
    assert plan_waves(tasks, max_parallel=10) == [["t1", "t2", "t4"], ["t3"], ["t5"]]


def test_done_tasks_are_satisfied_blockers_and_never_scheduled() -> None:
    tasks = [task("t1", status="done"), task("t2", blocked_by=["t1"])]
    assert plan_waves(tasks, max_parallel=4) == [["t2"]]
    assert plan_waves([task("t1", status="done")], max_parallel=4) == []


def test_a_cycle_fails_loudly() -> None:
    tasks = [task("t1", blocked_by=["t2"]), task("t2", blocked_by=["t1"])]
    with pytest.raises(DependencyCycleError, match="t1"):
        plan_waves(tasks, max_parallel=4)
    with pytest.raises(ValueError, match="max_parallel"):
        plan_waves([task("t1")], max_parallel=0)


def test_tasks_json_reads_loop_spec_field_names(tmp_path: Path) -> None:
    assert read_tasks(tmp_path, "s") == []
    feature = tmp_path / ".loop-spec" / "features" / "s"
    feature.mkdir(parents=True)
    (feature / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "task-001",
                    "subject": "scaffold",
                    "files": ["a.py"],
                    "blockedBy": [],
                    "acceptanceCriteria": ["a.py exists"],
                    "verifyCommand": "test -f a.py",
                    "readFirst": ["README.md"],
                    "status": "done",
                },
                {"not": "a task"},
            ]
        )
    )
    (only,) = read_tasks(tmp_path, "s")
    assert only.verify_command == "test -f a.py" and only.read_first == ["README.md"]
    assert only.acceptance_criteria == ["a.py exists"] and only.done
