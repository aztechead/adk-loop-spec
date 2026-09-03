# adk-loop-spec

An exemplar Google ADK (Python 3.14) application: a **dev-team assistant** that
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
│ intake (LLM) ──▶ route_request (deterministic code)                  │
│  {category,        │ CHANGE  (feature / bug)                          │
│   request}         │   └──▶ loop-spec working agent                   │
│                    │        SPEC → … → DELIVER → PR                   │
│                    │ QUESTION                                         │
│                    │   └──▶ qa agent: Memory Bank + A2A peer teams    │
│                    │ default ──▶ clarify (human input) ──▶ intake     │
└──────────────────────────────────────────────────────────────────────┘
```

## What it demonstrates

| Capability | Where |
|---|---|
| Graph-based workflow with routes, a back-edge, and a human-input node | `src/devteam/graph.py` |
| LiteLLM models: Gemini + Claude, API-key or Agent Platform, with per-model extras | `src/devteam/models.py`, `config/devteam.yaml` |
| Sessions + Memory Bank, in-memory or Agent Platform, incremental commits | `src/devteam/services.py`, `src/devteam/app.py` |
| A2A: bearer-protected exposure, peers with the A2A extension | `src/devteam/a2a.py`, `src/devteam/agents.py` |
| App lifecycle: resumable invocations, reflect-and-retry on tool failure | `src/devteam/app.py` |
| loop-spec mounted as the change-shipping engine | `src/devteam/loopspec.py` |
| Per-phase / per-role loop-spec model routing | `loop_spec.phases` / `loop_spec.roles` in the YAML |
| loop-spec's supervisor interface: oracle policy, store, sink, lifecycle | `src/devteam/supervisor.py`, `src/devteam/runtime.py` |

## Quickstart

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
git clone --recurse-submodules https://github.com/aztechead/adk-loop-spec
cd adk-loop-spec
uv sync

uv run devteam check          # validate config; prints every resolved route
uv run pytest                 # offline suite: scripted-LLM graph run + real local A2A round trip
uv run pyright                # the code is typed end to end; CI enforces it
```

Live runs need credentials for whatever `config/devteam.yaml` selects. Put
them in the environment or in a `.env` file (loaded automatically):

```sh
export GOOGLE_API_KEY=...        # gemini + api-key entries
export ANTHROPIC_API_KEY=...     # anthropic + api-key entries
export DEVTEAM_A2A_TOKEN=...     # bearer token peers must present to `devteam serve`
gcloud auth application-default login   # any agent-platform entry
```

```sh
uv run devteam chat "How do we deploy this service?"      # one message through the graph
uv run devteam serve                                      # expose the qa agent over A2A
uv run devteam supervise "add a healthcheck endpoint"     # unattended loop-spec run → PR
```

## Configuration

Everything lives in [`config/devteam.yaml`](config/devteam.yaml); the schema is
[`src/devteam/config.py`](src/devteam/config.py). The environment carries
secrets only.

**Models.** A provider entry is `{provider, backend, model, extra}`;
`(provider, backend)` maps onto one LiteLLM prefix, and `extra` is passed
verbatim to LiteLLM, so switching Claude from Anthropic's API to Agent
Platform — or turning on adaptive thinking — is a YAML edit:

```yaml
claude:
  provider: anthropic
  backend: agent-platform     # was: api-key
  model: claude-opus-5        # current-generation Claude uses bare ids on Agent Platform
  extra:
    thinking: {type: adaptive}
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
against the configured Agent Engine. The `memory_commit` plugin sends each
turn's new events to whichever memory store is active (so Memory Bank extracts
from fresh material, never the whole session again), and the qa agent reads it
back with `preload_memory` / `load_memory`.

**A2A teams.** `devteam serve` publishes an agent card at
`/.well-known/agent-card.json` and requires `Authorization: Bearer
$DEVTEAM_A2A_TOKEN` on everything else; binding a non-loopback host without a
token is refused. Only the qa agent is exposed — the loop-spec working agent
holds an unsandboxed shell and never leaves the process. List other instances
under `a2a.peers` (each with the token variable to present) and the qa agent
gains one tool per peer, so questions flow to the team that owns the answer:

```yaml
a2a:
  expose: {host: 0.0.0.0, port: 8001, token_env: DEVTEAM_A2A_TOKEN}
  peers:
    - name: platform_team
      url: http://platform.internal:8001
      token_env: PLATFORM_TEAM_TOKEN
```

Peers are consumed with `use_legacy=False`, which activates ADK's A2A
extension (no duplicated messages or lost nested output in streaming).

## The loop-spec mount

loop-spec lives at `third_party/loop-spec` (git submodule) and is imported
directly by the app — features and bugs route into its working agent, which
runs the SPEC → DISCUSS → PLAN → EXECUTE → VERIFY → ITERATE → DELIVER cycle
and ships one verified PR. `devteam supervise` drives that cycle unattended
on loop-spec's four supervisor ports, each configured under
`loop_spec.supervisor`:

| Port | Where it lands |
|---|---|
| decision oracle | `oracle.halt_when` / `oracle.prefer` / `oracle.pins` answer `get_user_choice`; the default is loop-spec's own `(Recommended)` option |
| state store | `store_dir` names loop-spec's `store-mirror.sh` adapter so an ephemeral container survives death |
| event sink | `events_file` names loop-spec's `append-sink.sh` adapter; every cycle event line lands there |
| lifecycle | phase handoffs are re-issued in a fresh context, an interrupted invocation is resumed by id, and a run that dies before writing its result is reconciled with `cycle-reconcile.sh` |

Success is judged by `outcome` + `converged`, never `status` alone.

loop-spec's EXECUTE fleet rung launches `adk run` against a mounted agent
directory. Generate that machine-local mount once (it is gitignored) and
`devteam` exports `LOOP_SPEC_ADK_AGENT_DIR` whenever it exists:

```sh
bash scripts/mount-loop-spec.sh
uv run adk run adk_agents/loop_spec     # or adk web
```

## Not used, and why

- **ADK Agent Config (YAML agents)** is Gemini-only and experimental; this app's
  point is one LiteLLM path for every vendor, so agents stay in typed Python
  and only their settings live in YAML.
- **`ContextCacheConfig`** needs native Gemini requests; LiteLLM traffic does
  not go through it.

## Layout

```
config/devteam.yaml     the one config file
src/devteam/
  config.py             typed YAML schema + loader
  models.py             (provider, backend, extra) → LiteLLM id / ADK model
  services.py           session + memory service pair
  agents.py             intake classifier, qa agent, authenticated A2A peers
  graph.py              the request Workflow graph (routes, human input, back-edge)
  app.py                composition root: Apps, Runner, plugins, resumability
  runtime.py            one turn with get_user_choice answered by a YAML oracle policy
  a2a.py                A2A server: qa agent only, bearer-token middleware
  loopspec.py           loop-spec mount + LOOP_SPEC_* environment
  supervisor.py         unattended loop-spec runs on the four ports
  cli.py                devteam check | chat | serve | supervise
tests/                  offline suite: scripted LLM through the real graph,
                        token-protected localhost A2A round trip, port contracts
third_party/loop-spec   the loop-spec checkout (submodule)
```
