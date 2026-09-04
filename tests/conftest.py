"""Shared fixtures: every test runs offline with dummy credentials."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from devteam.config import AppConfig, load_config
from devteam.models import AgentModel

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def dummy_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model construction checks these; no test may reach a real endpoint.

    Agent Platform clients are built lazily, so a project name is all the
    default backend needs at build time; ADC is only consulted on a request.
    """
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "offline-project")
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test")


@pytest.fixture()
def config() -> AppConfig:
    """The shipped config file, so tests validate what users actually run."""
    return load_config(REPO_ROOT / "config" / "devteam.yaml")


def base_raw() -> dict[str, object]:
    """The smallest valid config mapping, for tests that vary one section."""
    return {
        "models": {
            "providers": {"gemini-pro": {"provider": "gemini", "model": "gemini-2.5-pro"}},
            "agents": {"intake": "gemini-pro", "qa": "gemini-pro", "manager": "gemini-pro"},
        }
    }


class ScriptedLlm(BaseLlm):
    """A model that replies from a script and records every request.

    ``script`` maps a substring of an agent's instruction to the text that
    agent answers with; ``requests`` keeps each LlmRequest so a test can
    assert what an agent was shown.
    """

    script: dict[str, str]
    requests: list[LlmRequest] = Field(default_factory=list)

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["scripted"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        self.requests.append(llm_request)
        instruction = str(llm_request.config.system_instruction or "") if llm_request.config else ""
        text = next((reply for key, reply in self.script.items() if key in instruction), "")
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            turn_complete=True,
        )

    def as_agent_model(self) -> AgentModel:
        """What ``model_for_agent`` returns, with this script in the model's seat."""
        return AgentModel(self, None)
