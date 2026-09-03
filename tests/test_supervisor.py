"""The supervisor's ports, each testable without a model or a live cycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.adk.events import Event
from google.genai import types

from devteam.supervisor import (
    CycleResult,
    Supervisor,
    answer_question,
    ensure_profile,
    read_last_result,
)


def test_oracle_takes_the_recommended_option() -> None:
    options = ["halt", "Ship it (Recommended)", "ask again"]
    assert answer_question(options) == "Ship it (Recommended)"


def test_oracle_defaults_to_the_first_option() -> None:
    assert answer_question(["a", "b"]) == "a"


def test_result_reading_fails_loudly_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="died before producing a result"):
        read_last_result(tmp_path)


def test_success_requires_outcome_and_convergence() -> None:
    good = CycleResult(outcome="delivered", converged=True, phase_reached="DELIVER", handoffs=2)
    stalled = CycleResult(outcome="delivered", converged=False, phase_reached="VERIFY", handoffs=2)
    failed = CycleResult(outcome="failed", converged=True, phase_reached="EXECUTE", handoffs=0)
    assert good.succeeded
    assert not stalled.succeeded
    assert not failed.succeeded


def test_profile_is_written_once_and_never_clobbered(tmp_path: Path) -> None:
    path = ensure_profile(tmp_path)
    written = json.loads(path.read_text())
    assert written["preset"] == "supervised"
    assert written["env"]["LOOP_SPEC_PHASE_HANDOFF"] == "1"

    path.write_text('{"preset": "interactive"}')
    ensure_profile(tmp_path)
    assert json.loads(path.read_text()) == {"preset": "interactive"}


def pending_choice_event(options: list[str]) -> Event:
    call = types.FunctionCall(id="call-1", name="get_user_choice", args={"options": options})
    return Event(
        author="loop_spec",
        long_running_tool_ids={"call-1"},
        content=types.Content(role="model", parts=[types.Part(function_call=call)]),
    )


def test_pending_question_gets_a_function_response() -> None:
    event = pending_choice_event(["Proceed (Recommended)", "halt"])
    reply = Supervisor._maybe_answer(event)
    assert reply is not None
    response = reply.parts[0].function_response
    assert response is not None
    assert response.id == "call-1"
    assert response.response == {"result": "Proceed (Recommended)"}


def test_ordinary_events_get_no_response() -> None:
    event = Event(author="loop_spec", content=types.Content(role="model", parts=[]))
    assert Supervisor._maybe_answer(event) is None


def test_empty_options_are_an_error() -> None:
    with pytest.raises(ValueError, match="carried no options"):
        Supervisor._maybe_answer(pending_choice_event([]))
