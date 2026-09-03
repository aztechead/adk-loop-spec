# adk-loop-spec

An exemplar Google ADK (Python) application: a **dev-team assistant** that
classifies incoming requests with a graph workflow, ships features and bug
fixes through the [loop-spec](https://github.com/aztechead/loop-spec) cycle,
answers questions from long-term memory, and collaborates with other deployed
instances over A2A. Every model — Gemini or Claude, vendor API or Google Cloud
Agent Platform (formerly Vertex AI) — is driven through LiteLLM and selected in
one YAML file.

```
user request
    │
    ▼
┌─────────────────────── devteam Workflow graph ───────────────────────┐
│ intake (LLM classifier) ──▶ route_request (deterministic code)       │
│                               │ CHANGE  (feature / bug)              │
│                               │   └──▶ loop-spec working agent       │
│                               │        SPEC → … → DELIVER → PR       │
│                               │ QUESTION                             │
│                               │   └──▶ qa agent                      │
│                               │        Memory Bank + A2A peer teams  │
│                               │ default ──▶ clarify                  │
└──────────────────────────────────────────────────────────────────────┘
```

## What it demonstrates

| Capability | Where |
|---|---|
| Graph-based workflow with routes | `src/devteam/graph.py` |
| LiteLLM models: Gemini + Claude, API-key or Agent Platform | `src/devteam/models.py`, `config/devteam.yaml` |
| Sessions + Memory Bank, in-memory or Agent Platform | `src/devteam/services.py` |
| A2A: exposing this instance and consuming peers | `src/devteam/a2a.py`, `src/devteam/agents.py` |
| loop-spec mounted as the change-shipping engine | `src/devteam/loopspec.py` |
| Per-phase / per-role loop-spec model routing | `loop_spec.phases` / `loop_spec.roles` in the YAML |
| The loop-spec supervisor interface (all four ports) | `src/devteam/supervisor.py` |

## Quickstart

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
git clone --recurse-submodules https://github.com/aztechead/adk-loop-spec
cd adk-loop-spec
uv sync

uv run devteam check          # validate config; prints every resolved model route
uv run pytest                 # the offline suite, including a local A2A round trip
```

Live runs need credentials for whatever `config/devteam.yaml` selects:

```sh
export GOOGLE_API_KEY=...     # gemini + api-key entries
export ANTHROPIC_API_KEY=...  # anthropic + api-key entries
gcloud auth application-default login   # any agent-platform entry
```

```sh
uv run devteam chat "How do we deploy this service?"      # one message through the graph
uv run devteam serve                                      # expose this instance over A2A
uv run devteam supervise "add a healthcheck endpoint"     # unattended loop-spec run → PR
```

## Configuration

Everything lives in [`config/devteam.yaml`](config/devteam.yaml); the schema is
[`src/devteam/config.py`](src/devteam/config.py). The environment carries
secrets only.

**Models.** A provider entry is `{provider, backend, model}`; `(provider,
backend)` maps onto one LiteLLM prefix, so switching Claude from Anthropic's
API to Agent Platform is a one-line edit:

```yaml
claude:
  provider: anthropic
  backend: agent-platform     # was: api-key
  model: claude-sonnet-4-5@20250929
```

**Per-phase loop-spec models.** `loop_spec.phases` and `loop_spec.roles` name
provider entries and are exported as loop-spec's documented
`LOOP_SPEC_PHASE_MODEL_<PHASE>` / `LOOP_SPEC_MODEL_<ROLE>` variables — e.g.
spec and plan on Claude, execute on Gemini:

```yaml
loop_spec:
  phases: {spec: claude, plan: claude, execute: gemini-pro}
  roles: {code_reviewer: claude}
```

**Sessions and memory.** `services.backend: in-memory` runs everything locally;
`agent-platform` uses `VertexAiSessionService` + `VertexAiMemoryBankService`
against the configured Agent Engine. The `memory_commit` plugin feeds every
finished conversation into whichever memory store is active, and the qa agent
reads it back with `preload_memory` / `load_memory`.

**A2A teams.** Each instance publishes an agent card at
`/.well-known/agent-card.json` (`devteam serve`). List other instances under
`a2a.peers` and the qa agent gains one tool per peer — deploy several
instances, point their peer lists at each other, and questions flow to the
team that owns the answer:

```yaml
a2a:
  expose: {host: 0.0.0.0, port: 8001}
  peers:
    - name: platform_team
      url: http://platform.internal:8001
```

## The loop-spec mount

loop-spec lives at `third_party/loop-spec` (git submodule) and is imported
directly by the app — features and bugs route into its working agent, which
runs the SPEC → DISCUSS → PLAN → EXECUTE → VERIFY → ITERATE → DELIVER cycle
and ships one verified PR. `devteam supervise` drives that cycle unattended,
implementing loop-spec's four supervisor ports on their ADK seams
(`src/devteam/supervisor.py`): it answers `get_user_choice` interview
questions, resumes across phase handoffs, and judges the terminal result by
`outcome` + `converged`, never `status` alone.

To drive the loop-spec agents from the ADK CLI instead
(`adk run` / `adk web`), generate the machine-local mount:

```sh
bash scripts/mount-loop-spec.sh
uv run adk run adk_agents/loop_spec
```

## Layout

```
config/devteam.yaml     the one config file
src/devteam/
  config.py             typed YAML schema + loader
  models.py             (provider, backend) → LiteLLM id / ADK model
  services.py           session + memory service pair
  agents.py             intake classifier, qa agent, A2A peers
  graph.py              the request Workflow graph
  loopspec.py           loop-spec mount + phase/role model routing
  app.py                composition root: App, Runner, memory plugin
  a2a.py                A2A server (agent card + protocol endpoint)
  supervisor.py         unattended loop-spec runs (four ports)
  cli.py                devteam check | chat | serve | supervise
tests/                  offline suite; test_a2a_roundtrip.py is a real
                        localhost expose/consume round trip
third_party/loop-spec   the loop-spec checkout (submodule)
```
