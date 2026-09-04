"""The dev-team request graph: classify, route, act — locally or on a peer.

The whole control flow is a declared ADK ``Workflow`` graph — the routing
decision is deterministic code, never a model judgment:

    START -> intake (LLM) -> route_request -+-> PEER:<name> -> that team's instance (A2A)
                                            +-> CHANGE      -> manager loop over loop-spec
                                            +-> QUESTION    -> qa agent
                                            +-> default     -> clarify (human input)
                                                                 |
                                                     typed reply, back to route_request

Every deployed instance runs this same graph, so a request that belongs to
another team's repository is forwarded whole to that team's instance over
A2A, where its own intake classifies it and its own engineer ships it.
Feature and bug work for this team lands on the manager loop
(:mod:`devteam.manager`), which drives the mounted loop-spec working agent
one phase at a time from SPEC through DELIVER.
"""

from collections.abc import Callable, Iterator

from google.adk import Event, Workflow
from google.adk.agents import Context
from google.adk.events import EventActions, RequestInput
from google.adk.workflow import DEFAULT_ROUTE, BaseNode
from pydantic import BaseModel, Field

from devteam.agents import (
    INTAKE_STATE_KEY,
    Category,
    IntakeResult,
    build_intake_agent,
    build_peer_agent,
    build_qa_agent,
)
from devteam.config import AppConfig

# One edge per target: both change-shaped labels share the loop-spec route.
CHANGE_ROUTE = "CHANGE"
QUESTION_ROUTE = Category.QUESTION.value
PEER_ROUTE_PREFIX = "PEER:"

type NodeInput = IntakeResult | dict[str, object] | str | None


class Clarification(BaseModel):
    """What the human answers when intake could not classify: the label, and the request."""

    category: Category = Field(description="What you want: FEATURE, BUG, or QUESTION.")
    request: str = Field(description="The request in your own words.")


def peer_route(name: str) -> str:
    return f"{PEER_ROUTE_PREFIX}{name}"


def route_for(verdict: IntakeResult, peers: frozenset[str]) -> str:
    """A known peer team wins; otherwise the category decides."""
    if verdict.team and verdict.team in peers:
        return peer_route(verdict.team)
    match verdict.category:
        case Category.FEATURE | Category.BUG:
            return CHANGE_ROUTE
        case Category.QUESTION:
            return QUESTION_ROUTE


def parse_verdict(node_input: NodeInput) -> IntakeResult | None:
    """The verdict in whichever shape the graph hands it over, or None.

    A :class:`Clarification` from the human-input node is a verdict too: it
    carries the same two fields and no team.
    """
    match node_input:
        case IntakeResult():
            return node_input
        case dict():
            try:
                return IntakeResult.model_validate(node_input)
            except ValueError:
                return None
        case str():
            try:
                return IntakeResult.model_validate_json(node_input)
            except ValueError:
                return None
        case None:
            return None


def decide(node_input: NodeInput, stored: object, peers: frozenset[str]) -> Event:
    """The routing decision, pure: the node input first, the stored verdict second.

    The next node receives the event's ``output`` as its user turn, so the
    request — not the label — is what flows on. When neither the input nor the
    verdict intake wrote to session state parses, the default route carries the
    raw text for the clarify node to show.
    """
    verdict = parse_verdict(node_input)
    if verdict is None and isinstance(stored, dict | str):
        verdict = parse_verdict(stored)
    if verdict is None:
        return Event(actions=EventActions(route=[DEFAULT_ROUTE]), output=node_input)
    return Event(actions=EventActions(route=[route_for(verdict, peers)]), output=verdict.request)


def make_router(peers: frozenset[str]) -> Callable[[Context, NodeInput], Event]:
    """The routing node, closed over the peer names this instance knows."""

    def route_request(ctx: Context, node_input: NodeInput) -> Event:
        return decide(node_input, ctx.state.get(INTAKE_STATE_KEY), peers)

    return route_request


def clarify(node_input: NodeInput) -> Iterator[RequestInput]:
    """Human-input node: pause until the user answers with a typed clarification."""
    yield RequestInput(
        response_schema=Clarification,
        message=(
            f"I couldn't classify that request (the classifier produced {node_input!r}). "
            "Tell me whether you want a feature built, a bug fixed, or a question answered, "
            "and restate the request."
        ),
    )


def build_graph(config: AppConfig, change_node: BaseNode) -> Workflow:
    """The root workflow; the caller supplies the node that ships changes."""
    peers = {peer.name: build_peer_agent(peer) for peer in config.a2a.peers}
    route_request = make_router(frozenset(peers))
    intake = build_intake_agent(config)
    return Workflow(
        name=f"{config.app.name}_workflow",
        description=(
            "Dev-team assistant: classifies a request, ships features and bugs through "
            "loop-spec on this team's repository, answers questions from memory, and "
            "forwards work that belongs to a peer team."
        ),
        edges=[
            ("START", intake, route_request),
            (
                route_request,
                {
                    CHANGE_ROUTE: change_node,
                    QUESTION_ROUTE: build_qa_agent(config),
                    DEFAULT_ROUTE: clarify,
                    **{peer_route(name): agent for name, agent in peers.items()},
                },
            ),
            (clarify, route_request),
        ],
    )
