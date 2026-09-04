"""The manager loop: drive loop-spec one phase at a time, and judge between phases.

A long-horizon run tends to asymptote: the agent gets far, then stalls in the
minutiae. The remedy is a manager that hands the implementer one phase at a
time, checks the checklist between phases, and moves the cycle on when no box
has been ticked for a while. This module is that pattern as one native ADK
``Workflow`` graph over loop-spec's own phases (SPEC, DISCUSS, PLAN, EXECUTE,
VERIFY, ITERATE, DELIVER):

    START -> brief -> implementer -> observe -+-> JUDGE -> manager -> route_round -+-> CONTINUE -> implementer
                          ^                  |                                    +-> ASK -> ask_human -> resume_or_stop
                          |                  |                                    |                          |  |
                          +------------------+------------------------------------+--------------------------+  v
                                             +-> DONE ----------------------------------------------------> finish

- ``implementer`` is the mounted loop-spec working agent. As a workflow node
  it runs single-turn with no prior contents, so every round is a fresh
  context — the "separate thread" the pattern calls for. loop-spec's
  ``supervised`` profile makes it return after each durable phase with a
  paused ``phase-handoff`` result.
- ``observe`` is deterministic code: it reads loop-spec's result and PLAN
  checklist, appends a tick count to the progress ledger in session state,
  and flags a stall (no new tick for ``stall_rounds`` rounds). When PLAN has
  just handed off and ``manager.parallel`` is on, it detours through the
  fan-out nodes (:mod:`devteam.fanout`): plan waves, run each wave of
  implementers at once, integrate, repeat, then judge.
- ``manager`` is an LlmAgent with a typed verdict: continue, move on, or halt,
  plus one paragraph of guidance the next hand-off carries.
- ``route_round`` is code again: a terminal result ends the loop, a halt or an
  exhausted round budget asks a human, anything else re-issues the cycle.

The progress ledger doubles as the checklist page ``devteam serve`` renders.
"""

import datetime as dt
import enum
import html
import json
from collections.abc import Iterator
from pathlib import Path

from google.adk import Event, Workflow
from google.adk.agents import Context, LlmAgent
from google.adk.events import EventActions, RequestInput
from google.adk.workflow import BaseNode
from google.genai import types
from pydantic import BaseModel, Field

from devteam.config import AgentRole, AppConfig
from devteam.cycle import Checklist, CycleResult, read_checklist, read_last_result
from devteam.fanout import (
    RUN_ROUTE,
    WAVE_PLAN_KEY,
    ImplementerFactory,
    WaveSummary,
    build_wave_nodes,
    prune_outstanding_worktrees,
)
from devteam.models import model_for_agent
from devteam.waves import WavePlan

AUTO_PROMPT = "Load the loop-spec auto skill and run: {task}"
RESUME_PROMPT = "Load the loop-spec cycle skill and run: autonomous"

CONTINUE_ROUTE = "CONTINUE"
JUDGE_ROUTE = "JUDGE"
WAVES_ROUTE = "WAVES"
ASK_ROUTE = "ASK"
DONE_ROUTE = "DONE"
PLAN_PHASE = "plan"

# Session-state keys; the checklist page and the supervisor read them back.
ROUND_KEY = "manager_round"
LEDGER_KEY = "manager_ledger"
REPORT_KEY = "manager_report"
VERDICT_KEY = "manager_verdict"
RESULT_KEY = "manager_result"


class Decision(enum.StrEnum):
    """What the manager may decide between rounds."""

    CONTINUE = "CONTINUE"  # re-issue the cycle with guidance
    MOVE_ON = "MOVE_ON"  # re-issue it, telling the implementer to close the phase now
    HALT = "HALT"  # stop and ask a human


class PhaseVerdict(BaseModel):
    """The manager's typed judgment of one round."""

    decision: Decision = Field(description="CONTINUE, MOVE_ON, or HALT.")
    guidance: str = Field(
        default="", description="One short paragraph the next hand-off carries verbatim."
    )


class LedgerEntry(BaseModel):
    """One row of the progress ledger: how many boxes were ticked after a round."""

    round: int
    phase: str
    ticked: int
    total: int
    at: str  # ISO-8601, UTC; a string so the ledger round-trips through session state as JSON


class Ledger(BaseModel):
    """Ticks over time; the stall rule reads it."""

    entries: list[LedgerEntry] = Field(default_factory=list)

    def rounds_since_progress(self) -> int:
        """Trailing rounds in which neither the tick count rose nor the phase changed."""
        quiet = 0
        for previous, current in zip(reversed(self.entries[:-1]), reversed(self.entries)):
            if current.ticked > previous.ticked or current.phase != previous.phase:
                break
            quiet += 1
        return quiet


class RoundReport(BaseModel):
    """What the manager is shown after each round."""

    round: int
    result: CycleResult
    checklist: Checklist
    ledger: Ledger
    stalled: bool
    exhausted: bool
    waves: WaveSummary | None = Field(
        default=None, description="What the parallel fan-out merged or blocked after PLAN."
    )


class HumanDecision(BaseModel):
    """What a human answers when the loop halts."""

    resume: bool = Field(description="True to continue the cycle, false to stop here.")
    guidance: str = Field(default="", description="What to tell the implementer, if resuming.")


def parse_verdict(node_input: object) -> PhaseVerdict:
    """The manager's verdict in whichever shape the graph hands it over; CONTINUE when unusable."""
    match node_input:
        case PhaseVerdict():
            return node_input
        case dict():
            try:
                return PhaseVerdict.model_validate(node_input)
            except ValueError:
                return PhaseVerdict(decision=Decision.CONTINUE)
        case str():
            try:
                return PhaseVerdict.model_validate_json(node_input)
            except ValueError:
                return PhaseVerdict(decision=Decision.CONTINUE)
        case _:
            return PhaseVerdict(decision=Decision.CONTINUE)


def build_manager_agent(config: AppConfig) -> LlmAgent:
    """The manager: reads a round report, returns a :class:`PhaseVerdict`."""
    model = model_for_agent(AgentRole.MANAGER, config)
    return LlmAgent(
        name="manager",
        model=model.llm,
        generate_content_config=model.generation,
        description="Judges each loop-spec phase hand-off and steers the next one.",
        instruction=(
            "You manage an implementer that runs a loop-spec cycle one phase at a time.\n"
            "You receive a round report: the cycle result after the last phase, the PLAN "
            "checklist with which tasks are done, the progress ledger (ticks per round), "
            "and whether the run is stalled or out of rounds. After PLAN the report may "
            "carry waves: which tasks parallel implementers merged and which were blocked.\n"
            "Decide CONTINUE when the phase advanced or boxes were ticked. "
            "Decide MOVE_ON when stalled is true: the implementer must close the current "
            "phase with what it has instead of polishing. "
            "Decide HALT only when the result reports a failure a human must see, or when "
            "the same phase repeats with no change after a MOVE_ON.\n"
            "Write guidance as one short paragraph of concrete direction for the next "
            "phase. Ask for it to be done extremely well, never perfectly."
        ),
        output_schema=PhaseVerdict,
        output_key=VERDICT_KEY,
    )


def next_prompt(config: AppConfig, verdict: PhaseVerdict, stalled: bool) -> str:
    """The hand-off the implementer gets next: resume, guidance, then the phase prompt."""
    lines = [RESUME_PROMPT]
    if verdict.guidance:
        lines.append(f"Manager guidance: {verdict.guidance}")
    if verdict.decision is Decision.MOVE_ON or stalled:
        lines.append(
            "No checklist box has been ticked for several rounds: close the current "
            "phase with what you have and hand off. Do not polish."
        )
    lines.append(config.loop_spec.manager.phase_prompt)
    return "\n".join(lines)


def task_text(node_input: object) -> str:
    """The task as text, whether it arrived as the user's Content, a string, or a verdict."""
    match node_input:
        case types.Content():
            return "\n".join(part.text for part in node_input.parts or [] if part.text)
        case str():
            return node_input
        case _:
            return json.dumps(node_input, default=str)


def halted_result(reason: str) -> CycleResult:
    return CycleResult(status="paused", outcome="halted", reason=reason, converged=False)


def build_manager_loop(
    config: AppConfig,
    project_dir: Path,
    implementer: BaseNode,
    implementer_factory: ImplementerFactory | None = None,
) -> Workflow:
    """The manager loop over ``implementer`` (the mounted loop-spec working agent).

    ``implementer_factory`` overrides how parallel task implementers are built;
    tests use it to stand in for the model-backed ones.
    """
    manager_config = config.loop_spec.manager
    loop_spec_root = config.loop_spec.root
    waves = build_wave_nodes(config, project_dir, RoundReport, REPORT_KEY, implementer_factory)

    def brief(ctx: Context, node_input: object) -> str:
        """Round zero: reset the ledger and route the task into loop-spec."""
        task = task_text(node_input)
        ctx.state[ROUND_KEY] = 0
        ctx.state[LEDGER_KEY] = Ledger().model_dump(mode="json")
        ctx.state[REPORT_KEY] = None
        ctx.state[RESULT_KEY] = None
        return f"{AUTO_PROMPT.format(task=task)}\n{manager_config.phase_prompt}"

    def observe(ctx: Context) -> Event:
        """After a round: read loop-spec's result and checklist, extend the ledger.

        A terminal result skips the manager and ends the loop; a hand-off is
        handed to the manager as a :class:`RoundReport`.
        """
        result = read_last_result(loop_spec_root, project_dir)
        checklist = read_checklist(project_dir, result.slug)
        round_no = int(ctx.state.get(ROUND_KEY) or 0) + 1
        ledger = Ledger.model_validate(ctx.state.get(LEDGER_KEY) or {})
        ledger.entries.append(
            LedgerEntry(
                round=round_no,
                phase=result.phase_reached or "unknown",
                ticked=checklist.ticked,
                total=checklist.total,
                at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            )
        )
        report = RoundReport(
            round=round_no,
            result=result.model_copy(update={"handoffs": round_no - 1}),
            checklist=checklist,
            ledger=ledger,
            stalled=ledger.rounds_since_progress() >= manager_config.stall_rounds,
            exhausted=round_no >= manager_config.max_rounds,
        )
        ctx.state[ROUND_KEY] = round_no
        ctx.state[LEDGER_KEY] = ledger.model_dump(mode="json")
        ctx.state[REPORT_KEY] = report.model_dump(mode="json")
        if not result.is_handoff:
            return Event(
                actions=EventActions(route=[DONE_ROUTE]),
                output=report.result.model_dump(mode="json", by_alias=True),
            )
        fan_out = manager_config.parallel.enabled and result.handed_off_after(PLAN_PHASE)
        return Event(
            actions=EventActions(route=[WAVES_ROUTE if fan_out else JUDGE_ROUTE]),
            output=report.model_dump(mode="json"),
        )

    def route_round(ctx: Context, node_input: object) -> Event:
        """A halt or an exhausted round budget asks a human; anything else re-issues the cycle."""
        verdict = parse_verdict(node_input)
        report = RoundReport.model_validate(ctx.state[REPORT_KEY])
        if verdict.decision is Decision.HALT or report.exhausted:
            why = (
                f"the manager halted: {verdict.guidance or 'no reason given'}"
                if verdict.decision is Decision.HALT
                else f"round budget of {manager_config.max_rounds} is spent"
            )
            return Event(
                actions=EventActions(route=[ASK_ROUTE]),
                output=(
                    f"The cycle paused after phase {report.result.phase_reached!r} because "
                    f"{why}. Checklist: {report.checklist.ticked} of {report.checklist.total} "
                    "ticked. Resume with guidance, or stop here?"
                ),
            )
        return Event(
            actions=EventActions(route=[CONTINUE_ROUTE]),
            output=next_prompt(config, verdict, report.stalled),
        )

    def ask_human(node_input: str) -> Iterator[RequestInput]:
        """Human-input node: the loop waits here until someone decides."""
        yield RequestInput(response_schema=HumanDecision, message=node_input)

    def resume_or_stop(ctx: Context, node_input: object) -> Event:
        """A resume grants a fresh round budget; a stop ends the loop with a halted result."""
        decision = HumanDecision.model_validate(node_input)
        if decision.resume:
            ctx.state[ROUND_KEY] = 0
            verdict = PhaseVerdict(decision=Decision.CONTINUE, guidance=decision.guidance)
            return Event(
                actions=EventActions(route=[CONTINUE_ROUTE]),
                output=next_prompt(config, verdict, stalled=False),
            )
        if plan := ctx.state.get(WAVE_PLAN_KEY):
            # Nothing will resume these; leave the branches, drop the checkouts.
            prune_outstanding_worktrees(project_dir, WavePlan.model_validate(plan))
        return Event(
            actions=EventActions(route=[DONE_ROUTE]),
            output=halted_result("a human stopped the manager loop").model_dump(
                mode="json", by_alias=True
            ),
        )

    def finish(ctx: Context, node_input: object) -> dict[str, object]:
        """The one terminal node: record the result in state and hand it out."""
        result = CycleResult.model_validate(node_input)
        ctx.state[RESULT_KEY] = result.model_dump(mode="json", by_alias=True)
        return result.model_dump(mode="json", by_alias=True)

    manager = build_manager_agent(config)
    return Workflow(
        name="manager_loop",
        description=(
            "Ships one change through loop-spec phase by phase: a manager judges every "
            "phase hand-off, tracks the PLAN checklist, and moves the cycle on when it stalls."
        ),
        edges=[
            ("START", brief),
            (brief, implementer),
            (implementer, observe),
            (observe, {JUDGE_ROUTE: manager, WAVES_ROUTE: waves.plan, DONE_ROUTE: finish}),
            (waves.plan, {RUN_ROUTE: waves.execute, JUDGE_ROUTE: manager}),
            (waves.execute, waves.integrate),
            (waves.integrate, {RUN_ROUTE: waves.execute, JUDGE_ROUTE: manager}),
            (manager, route_round),
            (route_round, {CONTINUE_ROUTE: implementer, ASK_ROUTE: ask_human, DONE_ROUTE: finish}),
            (ask_human, resume_or_stop),
            (resume_or_stop, {CONTINUE_ROUTE: implementer, DONE_ROUTE: finish}),
        ],
    )


def render_progress(report: RoundReport | None, result: CycleResult | None) -> str:
    """The checklist page: boxes, a counter, and ticks over time, as one HTML document."""
    if report is None:
        body = "<p>No round has completed yet.</p>"
    else:
        checklist, ledger = report.checklist, report.ledger
        boxes = "".join(
            f"<li>{'&#9745;' if item.done else '&#9744;'} {html.escape(item.id)} "
            f"{html.escape(item.subject)}</li>"
            for item in checklist.items
        )
        rows = "".join(
            f"<tr><td>{e.round}</td><td>{html.escape(e.phase)}</td>"
            f"<td>{e.ticked}/{e.total}</td><td>{html.escape(e.at)}</td></tr>"
            for e in ledger.entries
        )
        points = " ".join(
            f"{i * 40 + 10},{110 - (e.ticked * 100 // e.total if e.total else 0)}"
            for i, e in enumerate(ledger.entries)
        )
        status = (
            f"finished: outcome={html.escape(result.outcome)} converged={result.converged}"
            if result
            else f"round {report.round}, phase {html.escape(report.result.phase_reached or '?')}"
            + (", stalled" if report.stalled else "")
        )
        waves = ""
        if report.waves is not None:
            blocked = ", ".join(
                f"{b.task_id} ({html.escape(b.reason)})" for b in report.waves.blocked
            )
            waves = (
                f"<p>Parallel waves: {report.waves.waves_run} of {report.waves.waves_total} run; "
                f"merged {', '.join(report.waves.merged) or 'none'}; "
                f"blocked {blocked or 'none'}.</p>"
            )
        body = (
            f"<p><strong>{checklist.ticked} of {checklist.total} boxes ticked</strong> ({status})</p>"
            f"{waves}"
            f"<ul>{boxes or '<li>PLAN has not written tasks yet.</li>'}</ul>"
            f"<svg width='{len(ledger.entries) * 40 + 20}' height='120' role='img' "
            "aria-label='ticks over rounds'><polyline fill='none' stroke='#2a7' "
            f"stroke-width='2' points='{points}'/></svg>"
            f"<table><tr><th>round</th><th>phase</th><th>ticked</th><th>at</th></tr>{rows}</table>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>devteam progress</title>"
        "<style>body{font-family:system-ui;margin:2rem}td,th{padding:0 .5rem}</style>"
        f"</head><body><h1>Manager loop progress</h1>{body}</body></html>"
    )
