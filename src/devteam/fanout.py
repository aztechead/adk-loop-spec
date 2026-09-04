"""Fan PLAN tasks out to parallel implementers, wave by wave, on native ADK.

After loop-spec's PLAN phase hands off, the manager loop takes the task list
and runs it as waves (:mod:`devteam.waves`). Each wave is one nested ADK
``Workflow``: START fans out to one implementer per task, a ``JoinNode`` waits
for all of them, and ``max_concurrency`` caps how many run at once. The wave
runs as a dynamic child of the manager loop through ``ctx.run_node``, so an
interrupted invocation resumes the wave instead of re-running it.

Each implementer is a plain LlmAgent on its own git worktree with ADK's
``EnvironmentToolset`` over a ``LocalEnvironment`` rooted there. It never
touches the loop-spec bridge, whose Execute tool reads the active skill
directory from one session-state key that concurrent agents would race on.
Integration is sequential and belongs to loop-spec: ``lib/integrate-task.sh``
rebases, verifies, and fast-forwards each task branch onto the feature
branch, then ``lib/task-progress.sh mark-done`` records it so loop-spec's
own EXECUTE phase never re-dispatches a published task.
"""

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from google.adk import Event, Workflow
from google.adk.agents import Context, LlmAgent
from google.adk.environment import LocalEnvironment
from google.adk.events import EventActions
from google.adk.tools.environment import EnvironmentToolset
from google.adk.workflow import BaseNode, FunctionNode, JoinNode
from pydantic import BaseModel, Field

from devteam.config import AppConfig
from devteam.cycle import LOOP_SPEC_DIR, CycleResult, run_script
from devteam.models import AgentModel, build_model
from devteam.waves import (
    DependencyCycleError,
    PlanTask,
    WavePlan,
    plan_waves,
    read_tasks,
    tasks_path,
)

RUN_ROUTE = "RUN"
JUDGE_ROUTE = "JUDGE"

WAVE_PLAN_KEY = "manager_wave_plan"
WAVE_SUMMARY_KEY = "manager_wave_summary"

INTEGRATE_TASK = Path("lib/integrate-task.sh")
TASK_PROGRESS = Path("lib/task-progress.sh")
WORKTREES_DIR = LOOP_SPEC_DIR / "worktrees"  # loop-spec's default base for task worktrees

type ImplementerFactory = Callable[[PlanTask, Path, str], BaseNode]
"""Builds the node that implements one task in one worktree on one branch."""


class TaskReport(BaseModel):
    """What an implementer returns; the same fields loop-spec's own template asks for."""

    task_id: str = Field(description="The task id from the brief, verbatim.")
    committed: bool = Field(description="True once the work is committed on the task branch.")
    sha: str = Field(default="", description="The commit sha, or empty when nothing was committed.")
    verify_passed: bool = Field(description="Whether the task's verify command passed.")
    notes: str = Field(default="", description="Concerns, or what blocked the task.")


class TaskOutcome(BaseModel):
    """What integration made of one task."""

    task_id: str
    status: Literal["merged", "blocked"]
    reason: str = ""
    sha: str = ""


class WaveSummary(BaseModel):
    """The fan-out's result so far, for the manager and the progress page."""

    slug: str
    waves_total: int = 0
    waves_run: int = 0
    merged: list[str] = Field(default_factory=list)
    blocked: list[TaskOutcome] = Field(default_factory=list)


def report_key(task_id: str) -> str:
    return f"task_report_{task_id}"


def node_name(task_id: str) -> str:
    return "implementer_" + re.sub(r"[^A-Za-z0-9_]", "_", task_id)


def task_branch(task_id: str, slug: str) -> str:
    return f"task/{task_id}-{slug}"


def worktree_path(project_dir: Path, task_id: str, slug: str) -> Path:
    return project_dir / WORKTREES_DIR / f"{task_id}-{slug}"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False
    )


def add_task_worktree(project_dir: Path, task_id: str, slug: str, feature_branch: str) -> Path:
    """The task's worktree on ``task/<id>-<slug>`` off the feature branch head; idempotent."""
    path = worktree_path(project_dir, task_id, slug)
    if path.is_dir():
        return path
    branch = task_branch(task_id, slug)
    exists = _git(project_dir, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    args = (
        ("worktree", "add", str(path), branch)
        if exists.returncode == 0
        else ("worktree", "add", "-b", branch, str(path), feature_branch)
    )
    added = _git(project_dir, *args)
    if added.returncode != 0:
        raise RuntimeError(f"git worktree add failed for {task_id}: {added.stderr.strip()}")
    return path


COMMITTER_NAME = "devteam"
COMMITTER_EMAIL = "devteam@localhost"


def committer_env(project_dir: Path) -> dict[str, str]:
    """A committer identity for the merge when the repository has none configured.

    The integrate script rebases task branches, and git refuses to rebase
    without a committer. Authorship stays with the implementer's commit; only
    the committer of the rebased copy is filled in, and only when the checkout
    (or the user's global config) does not already name one.
    """
    configured = _git(project_dir, "config", "user.email")
    if configured.returncode == 0 and configured.stdout.strip():
        return {}
    return {
        "GIT_COMMITTER_NAME": COMMITTER_NAME,
        "GIT_COMMITTER_EMAIL": COMMITTER_EMAIL,
        "GIT_AUTHOR_NAME": COMMITTER_NAME,
        "GIT_AUTHOR_EMAIL": COMMITTER_EMAIL,
    }


def integrate_task(
    loop_spec_root: Path,
    project_dir: Path,
    feature_branch: str,
    task: PlanTask,
    slug: str,
) -> TaskOutcome:
    """Rebase, verify, and fast-forward one task branch with loop-spec's own script."""
    result = run_script(
        loop_spec_root.resolve(),
        INTEGRATE_TASK,
        "--feature-root",
        str(project_dir),
        "--feature-branch",
        feature_branch,
        "--task-worktree",
        str(worktree_path(project_dir, task.id, slug)),
        "--task-branch",
        task_branch(task.id, slug),
        "--verify",
        task.verify_command,
        "--cleanup",
        cwd=project_dir,
        env=committer_env(project_dir),
    )
    try:
        record = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        record = {}
    if record.get("status") == "integrated" or record.get("published"):
        return TaskOutcome(task_id=task.id, status="merged", sha=str(record.get("candidate") or ""))
    reason = f"{record.get('reason') or 'integration failed'}: {record.get('detail') or ''}".strip(
        ": "
    )
    return TaskOutcome(task_id=task.id, status="blocked", reason=reason)


def mark_done(loop_spec_root: Path, project_dir: Path, slug: str, task_id: str) -> None:
    done = run_script(
        loop_spec_root.resolve(),
        TASK_PROGRESS,
        "mark-done",
        str(tasks_path(project_dir, slug)),
        task_id,
        cwd=project_dir,
    )
    if done.returncode != 0:
        raise RuntimeError(f"task-progress mark-done {task_id} failed: {done.stderr.strip()}")


IMPLEMENTER_INSTRUCTION = """You are an implementer agent for task {task_id}.

IMPORTANT: All paths must be ABSOLUTE. Do not use relative paths.
Engineering contract, read before writing code (never paste it):
{root}/skills/shared/implementer-contract.md, {root}/skills/shared/human-code.md,
{root}/skills/shared/writing-good-tests.md.

Step 1 - Your worktree already exists at {worktree} on branch {branch}.
  Do not run `git worktree add`. All file and git operations use that directory.
Step 2 - Read first: {read_first}. Then read the assigned files: {files}.
Step 3 - Implement the task in the worktree.
  Task subject: {subject}
  Acceptance criteria (every one binds):
{criteria}
  Touch ONLY the files listed. Do NOT edit unrelated files.
Step 4 - Run the task's verify command inside the worktree and make it pass:
  {verify}
Step 5 - Stage and commit inside the worktree branch:
  git -C "{worktree}" add <files>
  git -C "{worktree}" commit -m "feat: NO_JIRA {subject}"
  Do NOT push. Do NOT run git outside the worktree.

Do the task completely and extremely well, then report: task_id, committed,
the commit sha, whether verify passed, and any concerns in notes."""


def build_implementer(
    task: PlanTask, worktree: Path, branch: str, model: AgentModel, loop_spec_root: Path
) -> LlmAgent:
    """One implementer for one task: its own worktree, its own tools, a typed report."""
    root = loop_spec_root.resolve()
    environment = LocalEnvironment(
        working_dir=worktree,
        env_vars={"CLAUDE_PROJECT_DIR": str(worktree), "CLAUDE_PLUGIN_ROOT": str(root)},
    )
    return LlmAgent(
        name=node_name(task.id),
        model=model.llm,
        generate_content_config=model.generation,
        description=f"Implements PLAN task {task.id} in its own worktree.",
        instruction=IMPLEMENTER_INSTRUCTION.format(
            task_id=task.id,
            root=root,
            worktree=worktree,
            branch=branch,
            read_first=", ".join(task.read_first) or "nothing",
            files=", ".join(task.files) or "none listed",
            subject=task.subject,
            criteria="\n".join(f"  - {line}" for line in task.acceptance_criteria) or "  - none",
            verify=task.verify_command,
        ),
        include_contents="none",
        tools=[EnvironmentToolset(environment=environment)],
        output_schema=TaskReport,
        output_key=report_key(task.id),
    )


def build_wave_workflow(name: str, workers: list[BaseNode], max_concurrency: int) -> Workflow:
    """START fans out to every worker; the join waits for all of them."""
    join = JoinNode(name=f"{name}_join")
    return Workflow(
        name=name,
        max_concurrency=max_concurrency,
        edges=[("START", tuple(workers)), *((worker, join) for worker in workers)],
    )


class WaveNodes(BaseModel):
    """The three manager-loop nodes the fan-out contributes."""

    model_config = {"arbitrary_types_allowed": True}

    plan: Callable[..., object]
    execute: BaseNode  # a FunctionNode with rerun_on_resume, as ctx.run_node requires
    integrate: Callable[..., object]


def build_wave_nodes(
    config: AppConfig,
    project_dir: Path,
    report_model: type[BaseModel],
    report_key_name: str,
    implementer_factory: ImplementerFactory | None = None,
) -> WaveNodes:
    """The plan / execute / integrate nodes, closed over config and project.

    ``report_model`` and ``report_key_name`` are the manager loop's round
    report and its state key; the integrate node attaches the wave summary to
    it before the manager judges. ``implementer_factory`` defaults to
    :func:`build_implementer` on the configured model; tests inject a stub.
    """
    parallel = config.loop_spec.manager.parallel
    loop_spec_root = config.loop_spec.root
    factory = implementer_factory

    def make_worker(task: PlanTask, worktree: Path, branch: str) -> BaseNode:
        if factory is not None:
            return factory(task, worktree, branch)
        model = build_model(config.models.providers[config.loop_spec.implementer_key], config)
        return build_implementer(task, worktree, branch, model, loop_spec_root)

    def stored_report(ctx: Context) -> BaseModel:
        return report_model.model_validate(ctx.state[report_key_name])

    def judge(ctx: Context, report: BaseModel, summary: WaveSummary) -> Event:
        updated = report.model_copy(update={"waves": summary})
        ctx.state[report_key_name] = updated.model_dump(mode="json")
        ctx.state[WAVE_SUMMARY_KEY] = summary.model_dump(mode="json")
        return Event(
            actions=EventActions(route=[JUDGE_ROUTE]), output=updated.model_dump(mode="json")
        )

    def plan_waves_node(ctx: Context) -> Event:
        """Turn the PLAN task list into waves; nothing to run hands straight to the manager."""
        report = stored_report(ctx)
        result = cast(CycleResult, getattr(report, "result"))  # noqa: B009 - report_model is opaque here
        slug, branch = result.slug, result.feature_branch
        summary = WaveSummary(slug=slug or "")
        if not slug or not branch:
            return judge(ctx, report, summary)
        tasks = read_tasks(project_dir, slug)
        try:
            waves = plan_waves(tasks, parallel.max_parallel_implementers)
        except DependencyCycleError as error:
            summary.blocked.append(TaskOutcome(task_id="plan", status="blocked", reason=str(error)))
            return judge(ctx, report, summary)
        plan = WavePlan(slug=slug, branch=branch, waves=waves)
        summary.waves_total = len(waves)
        ctx.state[WAVE_PLAN_KEY] = plan.model_dump(mode="json")
        ctx.state[WAVE_SUMMARY_KEY] = summary.model_dump(mode="json")
        if plan.finished:
            return judge(ctx, report, summary)
        return Event(actions=EventActions(route=[RUN_ROUTE]), output=plan.current)

    async def execute_wave_node(ctx: Context) -> Event:
        """Run the current wave as one nested workflow: a worktree and an implementer per task."""
        plan = WavePlan.model_validate(ctx.state[WAVE_PLAN_KEY])
        tasks = {task.id: task for task in read_tasks(project_dir, plan.slug)}
        workers = [
            make_worker(
                tasks[task_id],
                add_task_worktree(project_dir, task_id, plan.slug, plan.branch),
                task_branch(task_id, plan.slug),
            )
            for task_id in plan.current
        ]
        wave = build_wave_workflow(
            f"wave_{plan.next_wave + 1}", workers, parallel.max_parallel_implementers
        )
        await ctx.run_node(
            wave, "Implement your assigned task.", run_id=f"wave-{plan.next_wave + 1}"
        )
        return Event(output=plan.current)

    def integrate_wave_node(ctx: Context) -> Event:
        """Merge the wave's committed tasks in order, mark them done, then run the next wave or judge."""
        plan = WavePlan.model_validate(ctx.state[WAVE_PLAN_KEY])
        summary = WaveSummary.model_validate(ctx.state[WAVE_SUMMARY_KEY])
        tasks = {task.id: task for task in read_tasks(project_dir, plan.slug)}
        for task_id in plan.current:
            raw = ctx.state.get(report_key(task_id))
            report = TaskReport.model_validate(raw) if isinstance(raw, dict) else None
            if report is None or not report.committed or not report.sha:
                why = (
                    report.notes if report and report.notes else "the implementer committed nothing"
                )
                summary.blocked.append(TaskOutcome(task_id=task_id, status="blocked", reason=why))
                continue
            outcome = integrate_task(
                loop_spec_root, project_dir, plan.branch, tasks[task_id], plan.slug
            )
            if outcome.status == "merged":
                mark_done(loop_spec_root, project_dir, plan.slug, task_id)
                summary.merged.append(task_id)
            else:
                summary.blocked.append(outcome)
        plan.next_wave += 1
        summary.waves_run = plan.next_wave
        ctx.state[WAVE_PLAN_KEY] = plan.model_dump(mode="json")
        ctx.state[WAVE_SUMMARY_KEY] = summary.model_dump(mode="json")
        if plan.finished:
            return judge(ctx, stored_report(ctx), summary)
        return Event(actions=EventActions(route=[RUN_ROUTE]), output=plan.current)

    return WaveNodes(
        plan=plan_waves_node,
        # A node that schedules children dynamically re-runs after an interrupt so
        # it can collect their results; ADK refuses ctx.run_node without this.
        execute=FunctionNode(func=execute_wave_node, name="execute_wave", rerun_on_resume=True),
        integrate=integrate_wave_node,
    )
