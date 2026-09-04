"""The parallel fan-out: real git worktrees, loop-spec's integrate script, waves in order."""

import json
import subprocess
from pathlib import Path

import pytest
from google.adk.agents import Context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import BaseNode, FunctionNode

from devteam.config import AppConfig
from devteam.cycle import CycleResult, read_checklist
from devteam.fanout import (
    WAVE_SUMMARY_KEY,
    TaskReport,
    WaveSummary,
    add_task_worktree,
    build_implementer,
    report_key,
    task_branch,
    worktree_path,
)
from devteam.manager import RESULT_KEY, Decision, PhaseVerdict, build_manager_loop
from devteam.models import AgentModel
from devteam.runtime import policy_oracle, run_turn, text_message
from devteam.waves import PlanTask
from tests.conftest import ScriptedLlm, base_raw

SLUG = "demo"
FEATURE_BRANCH = f"feat/{SLUG}"
GIT_IDENTITY = ["-c", "user.name=test", "-c", "user.email=test@example.com"]


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *GIT_IDENTITY, "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def tasks_json() -> list[dict[str, object]]:
    return [
        {"id": "t1", "subject": "write a", "files": ["a.txt"], "verifyCommand": "test -f a.txt"},
        {"id": "t2", "subject": "write b", "files": ["b.txt"], "verifyCommand": "test -f b.txt"},
        {
            "id": "t3",
            "subject": "extend a",
            "files": ["a.txt"],
            "blockedBy": ["t1"],
            "verifyCommand": "grep -q extended a.txt",
        },
    ]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A repository the way loop-spec leaves it after PLAN: on feat/<slug>, tasks.json written."""
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("demo\n")
    (tmp_path / ".gitignore").write_text(".loop-spec/\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    git(tmp_path, "checkout", "-q", "-b", FEATURE_BRANCH)
    feature = tmp_path / ".loop-spec" / "features" / SLUG
    feature.mkdir(parents=True)
    (feature / "tasks.json").write_text(json.dumps(tasks_json()))
    return tmp_path


def parallel_config(**parallel: object) -> AppConfig:
    raw = base_raw() | {
        "loop_spec": {"manager": {"parallel": {"enabled": True, **parallel}}},
    }
    return AppConfig.model_validate(raw)


class FakeImplementer:
    """Stands in for the model-backed implementer: edits, commits, reports, in its worktree."""

    def __init__(self, skip_commit: frozenset[str] = frozenset()) -> None:
        self.skip_commit = skip_commit
        self.seen: list[tuple[str, Path, str]] = []

    def __call__(self, task: PlanTask, worktree: Path, branch: str) -> BaseNode:
        self.seen.append((task.id, worktree, branch))

        def implement(ctx: Context) -> TaskReport:
            target = worktree / task.files[0]
            existing = target.read_text() if target.exists() else ""
            target.write_text(existing + ("extended\n" if task.id == "t3" else f"{task.id}\n"))
            if task.id in self.skip_commit:
                report = TaskReport(
                    task_id=task.id, committed=False, verify_passed=False, notes="gave up"
                )
            else:
                git(worktree, "add", task.files[0])
                git(worktree, "commit", "-q", "-m", f"feat: NO_JIRA {task.subject}")
                report = TaskReport(
                    task_id=task.id,
                    committed=True,
                    sha=git(worktree, "rev-parse", "HEAD"),
                    verify_passed=True,
                )
            ctx.state[report_key(task.id)] = report.model_dump()
            return report

        return FunctionNode(func=implement, name=f"implementer_{task.id}")


class FakeLoopSpec:
    """The loop-spec working agent: hands off after PLAN, then delivers."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.prompts: list[str] = []

    def loop_spec(self, ctx: Context, node_input: str) -> str:
        self.prompts.append(node_input)
        if len(self.prompts) == 1:
            result = {"status": "paused", "reason": "phase-handoff", "phaseReached": "plan"}
        else:
            result = {
                "status": "completed",
                "outcome": "delivered",
                "phaseReached": "deliver",
                "converged": True,
            }
        (self.project_dir / ".loop-spec" / "last-result.json").write_text(
            json.dumps(result | {"slug": SLUG, "branch": FEATURE_BRANCH})
        )
        return "phase done"


async def run_loop(
    repo: Path, config: AppConfig, implementers: FakeImplementer, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeLoopSpec, ScriptedLlm, dict[str, object]]:
    manager = ScriptedLlm(
        model="scripted",
        script={
            "You manage": PhaseVerdict(
                decision=Decision.CONTINUE, guidance="Go on."
            ).model_dump_json()
        },
    )
    monkeypatch.setattr(
        "devteam.manager.model_for_agent", lambda role, cfg: manager.as_agent_model()
    )
    fake = FakeLoopSpec(repo)
    workflow = build_manager_loop(
        config, repo, FunctionNode(func=fake.loop_spec, name="loop_spec"), implementers
    )
    runner = Runner(node=workflow, app_name="t", session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name="t", user_id="u")
    async for _ in run_turn(
        runner,
        user_id="u",
        session_id=session.id,
        message=text_message("build the demo"),
        oracle=policy_oracle(config.loop_spec.supervisor.oracle),
    ):
        pass
    stored = await runner.session_service.get_session(
        app_name="t", user_id="u", session_id=session.id
    )
    assert stored is not None
    return fake, manager, dict(stored.state)


async def test_waves_run_in_order_and_land_on_the_feature_branch(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementers = FakeImplementer()
    fake, manager, state = await run_loop(
        repo, parallel_config(max_parallel_implementers=2), implementers, monkeypatch
    )

    # Wave 1 held the two independent tasks, wave 2 the one that depended on t1.
    assert [seen[0] for seen in implementers.seen] == ["t1", "t2", "t3"]
    assert implementers.seen[0][2] == task_branch("t1", SLUG)
    summary = WaveSummary.model_validate(state[WAVE_SUMMARY_KEY])
    assert (summary.waves_total, summary.waves_run, summary.merged, summary.blocked) == (
        2,
        2,
        ["t1", "t2", "t3"],
        [],
    )

    # loop-spec's integrate script fast-forwarded every task branch, in order, and cleaned up.
    subjects = git(repo, "log", "--format=%s", FEATURE_BRANCH).splitlines()
    assert subjects == [
        "feat: NO_JIRA extend a",
        "feat: NO_JIRA write b",
        "feat: NO_JIRA write a",
        "init",
    ]
    assert (repo / "a.txt").read_text() == "t1\nextended\n"
    assert not worktree_path(repo, "t1", SLUG).exists()
    assert "task/t1-demo" not in git(repo, "branch", "--list")

    # Every task is marked done, so loop-spec's EXECUTE has nothing left to dispatch.
    checklist = read_checklist(repo, SLUG)
    assert (checklist.ticked, checklist.total) == (3, 3)

    # The manager saw the wave summary once, then the cycle resumed and delivered.
    assert len(manager.requests) == 1
    shown = manager.requests[0].contents[-1].parts
    assert shown and shown[0].text and '"merged": ["t1", "t2", "t3"]' in shown[0].text
    assert len(fake.prompts) == 2 and "Go on." in fake.prompts[1]
    assert CycleResult.model_validate(state[RESULT_KEY]).succeeded


async def test_an_uncommitted_task_is_blocked_not_merged(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementers = FakeImplementer(skip_commit=frozenset({"t2"}))
    _, _, state = await run_loop(repo, parallel_config(), implementers, monkeypatch)
    summary = WaveSummary.model_validate(state[WAVE_SUMMARY_KEY])
    assert summary.merged == ["t1", "t3"]
    assert [(b.task_id, b.reason) for b in summary.blocked] == [("t2", "gave up")]
    assert read_checklist(repo, SLUG).ticked == 2
    assert not (repo / "b.txt").exists()


def test_worktrees_branch_off_the_feature_head_and_are_reused(repo: Path) -> None:
    path = add_task_worktree(repo, "t1", SLUG, FEATURE_BRANCH)
    assert path == worktree_path(repo, "t1", SLUG) and (path / "README.md").is_file()
    assert git(path, "rev-parse", "--abbrev-ref", "HEAD") == task_branch("t1", SLUG)
    assert add_task_worktree(repo, "t1", SLUG, FEATURE_BRANCH) == path


def test_model_backed_implementer_is_isolated_and_typed(tmp_path: Path, config: AppConfig) -> None:
    llm = ScriptedLlm(model="scripted", script={})
    task = PlanTask.model_validate(
        {
            "id": "t1",
            "subject": "write a",
            "files": ["a.txt"],
            "acceptanceCriteria": ["a.txt says hi"],
            "verifyCommand": "test -f a.txt",
        }
    )
    agent = build_implementer(
        task, tmp_path, "task/t1-demo", AgentModel(llm, None), config.loop_spec.root
    )
    assert agent.name == "implementer_t1" and agent.include_contents == "none"
    assert agent.output_schema is TaskReport and agent.output_key == report_key("t1")
    instruction = str(agent.instruction)
    assert (
        str(tmp_path) in instruction
        and "test -f a.txt" in instruction
        and "a.txt says hi" in instruction
    )
