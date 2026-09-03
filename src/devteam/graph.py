"""The dev-team request graph: classify, route, act.

The whole control flow is a declared ADK ``Workflow`` graph — the routing
decision is deterministic code, never a model judgment:

    START -> intake (LLM) -> route_request -+-> FEATURE -> loop-spec agent
                                            +-> BUG     -> loop-spec agent
                                            +-> QUESTION-> qa agent
                                            +-> default -> clarify

Feature and bug work lands on the mounted loop-spec working agent, which runs
its own graph-driven cycle (SPEC through DELIVER) beneath this one.
"""

from __future__ import annotations

from google.adk import Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.workflow import DEFAULT_ROUTE

from .agents import Category, build_intake_agent, build_qa_agent
from .config import AppConfig

# One edge per target: both change-shaped labels share the loop-spec route.
CHANGE_ROUTE = "CHANGE"
QUESTION_ROUTE = Category.QUESTION.value


def route_request(node_input: str) -> Event:
    """Turn the classifier's label into a graph route, forwarding the label as output."""
    label = node_input.strip().upper()
    match label:
        case Category.FEATURE | Category.BUG:
            route = CHANGE_ROUTE
        case Category.QUESTION:
            route = QUESTION_ROUTE
        case _:
            route = DEFAULT_ROUTE
    return Event(route=[route], output=label)


def clarify(node_input: str) -> Event:
    """Terminal branch for a label the graph has no edge for."""
    return Event(
        message=(
            f"I couldn't classify that request (the classifier said {node_input!r}). "
            "Tell me whether you want a feature built, a bug fixed, or a question answered."
        )
    )


def build_graph(config: AppConfig, loop_spec_agent: LlmAgent) -> Workflow:
    """The root workflow; the caller supplies the mounted loop-spec agent."""
    qa_agent = build_qa_agent(config)
    return Workflow(
        name=f"{config.app.name}_workflow",
        description="Classifies dev-team requests and routes them to the right agent.",
        edges=[
            ("START", build_intake_agent(config), route_request),
            (
                route_request,
                {
                    CHANGE_ROUTE: loop_spec_agent,
                    QUESTION_ROUTE: qa_agent,
                    DEFAULT_ROUTE: clarify,
                },
            ),
        ],
    )
