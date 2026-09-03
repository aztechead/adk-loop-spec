"""The dev-team request graph: classify, route, act.

The whole control flow is a declared ADK ``Workflow`` graph — the routing
decision is deterministic code, never a model judgment:

    START -> intake (LLM) -> route_request -+-> CHANGE   -> loop-spec agent
                                            +-> QUESTION -> qa agent
                                            +-> default  -> clarify (human input)
                                                              |
                                                     back to intake

Feature and bug work lands on the mounted loop-spec working agent, which runs
its own graph-driven cycle (SPEC through DELIVER) beneath this one. A request
the classifier could not place pauses on a human-input node and re-enters
intake with the user's restatement.
"""

from collections.abc import Iterator

from google.adk import Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.events import EventActions, RequestInput
from google.adk.workflow import DEFAULT_ROUTE
from google.genai import types

from .agents import Category, IntakeResult, build_intake_agent, build_qa_agent
from .config import AppConfig

# One edge per target: both change-shaped labels share the loop-spec route.
CHANGE_ROUTE = "CHANGE"
QUESTION_ROUTE = Category.QUESTION.value


def route_for(category: Category) -> str:
    match category:
        case Category.FEATURE | Category.BUG:
            return CHANGE_ROUTE
        case Category.QUESTION:
            return QUESTION_ROUTE


def route_request(node_input: IntakeResult | dict[str, object] | str) -> Event:
    """Turn the classifier's verdict into a graph route, forwarding the request text.

    The next node receives this event's ``output`` as its user turn, so the
    request — not the label — is what flows on. An unparseable verdict takes
    the default route with the raw text, so the clarify node can show it.
    """
    match node_input:
        case IntakeResult():
            verdict = node_input
        case dict():
            verdict = IntakeResult.model_validate(node_input)
        case str():
            try:
                verdict = IntakeResult.model_validate_json(node_input)
            except ValueError:
                return Event(actions=EventActions(route=[DEFAULT_ROUTE]), output=node_input)
    return Event(actions=EventActions(route=[route_for(verdict.category)]), output=verdict.request)


def clarify(node_input: str) -> Iterator[RequestInput]:
    """Human-input node: pause until the user restates, then feed intake again."""
    yield RequestInput(
        response_schema=str,
        message=(
            f"I couldn't classify that request (the classifier produced {node_input!r}). "
            "Tell me whether you want a feature built, a bug fixed, or a question answered."
        ),
    )


def user_message(text: str) -> Event:
    """A user-facing text event for terminal function nodes."""
    return Event(content=types.Content(role="model", parts=[types.Part(text=text)]))


def build_graph(config: AppConfig, loop_spec_agent: LlmAgent) -> Workflow:
    """The root workflow; the caller supplies the mounted loop-spec agent."""
    intake = build_intake_agent(config)
    qa_agent = build_qa_agent(config)
    return Workflow(
        name=f"{config.app.name}_workflow",
        description="Classifies dev-team requests and routes them to the right agent.",
        edges=[
            ("START", intake, route_request),
            (
                route_request,
                {
                    CHANGE_ROUTE: loop_spec_agent,
                    QUESTION_ROUTE: qa_agent,
                    DEFAULT_ROUTE: clarify,
                },
            ),
            (clarify, intake),
        ],
    )
