"""Run one conversational turn, answering the agent's questions as they arrive.

loop-spec interviews through ADK's ``get_user_choice`` long-running tool. A
pending call surfaces as an event whose ``long_running_tool_ids`` names it; the
run cannot proceed until someone sends a function response. ``run_turn`` is
the one place that loop lives, so ``chat`` and ``supervise`` never diverge.
"""

from collections.abc import AsyncIterator, Callable

from google.adk.events import Event
from google.adk.runners import Runner
from google.genai import types

from .config import OraclePolicy

type Oracle = Callable[[list[str]], str]
"""Chooses one of the options a question offers (or ``halt``)."""

RECOMMENDED_MARK = "(Recommended)"
HALT = "halt"  # loop-spec's reserved answer: pause the cycle for a human
QUESTION_TOOL = "get_user_choice"


def policy_oracle(policy: OraclePolicy) -> Oracle:
    """The decision oracle loop-spec's supervisor port describes, driven by YAML.

    First rule wins: a ``halt_when`` match pauses the cycle, a ``prefer`` match
    picks that option, otherwise the ``(Recommended)`` option — which loop-spec
    records as an assumed self-answer, so only the first two add policy.
    """

    def choose(options: list[str]) -> str:
        if any(needle in option for needle in policy.halt_when for option in options):
            return HALT
        for needle in policy.prefer:
            for option in options:
                if needle in option:
                    return option
        return next((o for o in options if RECOMMENDED_MARK in o), options[0])

    return choose


def text_message(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def answer_for(event: Event, oracle: Oracle) -> types.Content | None:
    """The function response answering an event's pending question, if any."""
    if not event.long_running_tool_ids or event.content is None:
        return None
    for part in event.content.parts or []:
        call = part.function_call
        if call is None or call.name != QUESTION_TOOL or call.id not in event.long_running_tool_ids:
            continue
        options = list((call.args or {}).get("options", []))
        if not options:
            raise ValueError(f"{QUESTION_TOOL} call {call.id} carried no options")
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response={"result": oracle(options)}
                    )
                )
            ],
        )
    return None


async def run_turn(
    runner: Runner, *, user_id: str, session_id: str, message: types.Content, oracle: Oracle
) -> AsyncIterator[Event]:
    """Yield every event of a turn, resuming the run after each answered question."""
    pending: types.Content | None = message
    while pending is not None:
        reply: types.Content | None = None
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=pending
        ):
            yield event
            if (answer := answer_for(event, oracle)) is not None:
                reply = answer
        pending = reply
