"""The supervisor's ports, each testable without a model or a live cycle."""

import json
from pathlib import Path

import pytest
from google.adk.events import Event
from google.genai import types

from devteam.config import AppConfig, OraclePolicy
from devteam.cycle import CycleResult, read_last_result
from devteam.runtime import HALT, answer_for, policy_oracle
from devteam.supervisor import ensure_profile, profile_env
from tests.conftest import base_raw


def test_oracle_takes_the_recommended_option_by_default() -> None:
    choose = policy_oracle(OraclePolicy())
    assert choose(["halt", "Ship it (Recommended)", "ask again"]) == "Ship it (Recommended)"
    assert choose(["a", "b"]) == "a"


def test_oracle_policy_prefers_and_halts_from_yaml() -> None:
    choose = policy_oracle(OraclePolicy(prefer=("compact",), halt_when=("delete",)))
    assert choose(["Full cycle (Recommended)", "Run compact cycle"]) == "Run compact cycle"
    assert choose(["Keep (Recommended)", "delete the directory"]) == HALT


def test_success_requires_outcome_and_convergence() -> None:
    good = CycleResult(outcome="delivered", converged=True, phase_reached="DELIVER", handoffs=2)
    stalled = CycleResult(outcome="delivered", converged=False, phase_reached="VERIFY", handoffs=2)
    failed = CycleResult(outcome="failed", converged=True, phase_reached="EXECUTE", handoffs=0)
    assert good.succeeded
    assert not stalled.succeeded
    assert not failed.succeeded


def test_result_record_is_read_as_loop_spec_writes_it(config: AppConfig, tmp_path: Path) -> None:
    """The contract's camelCase record, including fields we do not use, parses into the model."""
    record = {
        "schema": 1,
        "status": "paused",
        "outcome": "in-progress",
        "reason": "phase-handoff",
        "summary": "PLAN closed",
        "slug": "healthcheck",
        "phaseReached": "plan",
        "converged": False,
        "workDelivered": False,
        "prUrl": None,
        "iterations": {"used": 0, "max": None},
    }
    (tmp_path / ".loop-spec").mkdir()
    (tmp_path / ".loop-spec" / "last-result.json").write_text(json.dumps(record))
    result = read_last_result(config.loop_spec.root, tmp_path)
    assert result.is_handoff and not result.succeeded
    assert (result.slug, result.phase_reached, result.handoffs) == ("healthcheck", "plan", 0)


def test_profile_names_the_store_and_sink_adapters(tmp_path: Path) -> None:
    raw = base_raw() | {
        "loop_spec": {
            "supervisor": {
                "store_dir": str(tmp_path / "mirror"),
                "events_file": str(tmp_path / "e.jsonl"),
            }
        }
    }
    env = profile_env(AppConfig.model_validate(raw))
    assert env["LOOP_SPEC_STORE"].endswith("lib/supervisor/store-mirror.sh")
    assert env["LOOP_SPEC_STORE_DIR"] == str(tmp_path / "mirror")
    assert env["LOOP_SPEC_EVENT_SINK"].endswith("examples/supervisor/append-sink.sh")
    assert env["LOOP_SPEC_EVENT_SINK_FILE"] == str(tmp_path / "e.jsonl")


def test_profile_is_written_validated_and_never_clobbered(
    config: AppConfig, tmp_path: Path
) -> None:
    path = ensure_profile(config, tmp_path)
    assert json.loads(path.read_text()) == {"preset": "supervised", "env": {}}

    path.write_text('{"preset": "interactive"}')
    ensure_profile(config, tmp_path)
    assert json.loads(path.read_text()) == {"preset": "interactive"}

    path.write_text('{"preset": "no-such-preset"}')
    with pytest.raises(RuntimeError, match="rejected"):
        ensure_profile(config, tmp_path)


def test_missing_result_is_reconciled_or_fails_loudly(config: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cycle-reconcile"):
        read_last_result(config.loop_spec.root, tmp_path)


def pending_choice_event(options: list[str]) -> Event:
    call = types.FunctionCall(id="call-1", name="get_user_choice", args={"options": options})
    return Event(
        author="loop_spec",
        long_running_tool_ids={"call-1"},
        content=types.Content(role="model", parts=[types.Part(function_call=call)]),
    )


def test_pending_question_gets_a_function_response() -> None:
    reply = answer_for(
        pending_choice_event(["Proceed (Recommended)", "halt"]), policy_oracle(OraclePolicy())
    )
    assert reply is not None and reply.parts
    response = reply.parts[0].function_response
    assert response is not None
    assert response.id == "call-1"
    assert response.response == {"result": "Proceed (Recommended)"}


def test_ordinary_events_get_no_response() -> None:
    event = Event(author="loop_spec", content=types.Content(role="model", parts=[]))
    assert answer_for(event, policy_oracle(OraclePolicy())) is None


def test_empty_options_are_an_error() -> None:
    with pytest.raises(ValueError, match="carried no options"):
        answer_for(pending_choice_event([]), policy_oracle(OraclePolicy()))
