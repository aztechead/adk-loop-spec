# adk-loop-spec

An exemplar Google ADK (Python 3.14) application: a **dev-team assistant** that
classifies incoming requests with a graph workflow, ships features and bug
fixes through the [loop-spec](https://github.com/aztechead/loop-spec) cycle
under a manager loop, answers questions from long-term memory, and
collaborates with other deployed instances over A2A. Every model runs on
Google Cloud Agent Platform (formerly Vertex AI) with Application Default
Credentials by default; the vendor APIs are an opt-in switch in one YAML file.

```
user request
    │
    ▼
┌─────────────────────── devteam Workflow graph ───────────────────────┐
│ intake (LLM, typed verdict) ──▶ route_request (deterministic code)   │
│                    │ CHANGE  (feature / bug)                          │
│                    │   └──▶ manager loop ──▶ loop-spec working agent  │
│                    │        one phase per round: SPEC → … → DELIVER  │
│                    │ QUESTION                                         │
│                    │   └──▶ qa agent: Memory Bank + A2A peer teams    │
│                    │ PEER:<team> ──▶ that team's instance over A2A    │
│                    │ default ──▶ clarify (typed human input) ──┐      │
│                    ◀──────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

## What it demonstrates

| Capability | Where |
|---|---|
| Graph-based workflow with routes, back-edges, and typed human-input nodes | `src/devteam/graph.py`, `src/devteam/manager.py` |
| Structured data handling: `output_schema` + `output_key` on every agent, typed `RequestInput` replies | `src/devteam/agents.py`, `src/devteam/graph.py`, `src/devteam/manager.py` |
| Native ADK models on Agent Platform with ADC; LiteLLM only for vendor API keys | `src/devteam/models.py`, `config/devteam.yaml` |
| Sessions + Memory Bank, in-memory or Agent Platform, incremental commits | `src/devteam/services.py`, `src/devteam/app.py` |
| A2A on FastAPI: the full graph, bearer auth, direct HTTPS, a progress page | `src/devteam/a2a.py` |
| The manager loop: one loop-spec phase per round, a checklist ledger, a stall rule | `src/devteam/manager.py` |
| App lifecycle: resumable invocations, reflect-and-retry on tool failure | `src/devteam/app.py` |
| loop-spec mounted as the change-shipping engine, per-phase / per-role model routing | `src/devteam/loopspec.py` |
| loop-spec's supervisor interface: oracle policy, store, sink, lifecycle | `src/devteam/supervisor.py`, `src/devteam/runtime.py` |

## Quickstart

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
git clone --recurse-submodules https://github.com/aztechead/adk-loop-spec
cd adk-loop-spec
uv sync

uv run devteam check          # validate config; prints every resolved route
uv run pytest                 # offline suite: scripted-LLM graph runs + real local A2A round trips
uv run pyright                # the code is typed end to end; CI enforces it
```

Live runs authenticate with ADC and need a project. Put these in the
environment or in a `.env` file (loaded automatically):

```sh
gcloud auth application-default login   # ADC: the default for every model, session, and memory call
export GOOGLE_CLOUD_PROJECT=...          # or set gcp.project in the YAML
export DEVTEAM_A2A_TOKEN=...             # bearer token peers must present to `devteam serve`
export GOOGLE_API_KEY=...                # only for a provider with backend: api-key
export ANTHROPIC_API_KEY=...             # only for a provider with backend: api-key
```

```sh
uv run devteam chat "How do we deploy this service?"      # one message through the graph
uv run devteam serve                                      # expose this instance over A2A
uv run devteam supervise "add a healthcheck endpoint"     # unattended manager loop → PR
```

## Code standards

- `__init__.py` files hold English comments only, never code.
- Imports are absolute (`from devteam.config import ...`); ruff bans relative imports.
- FastAPI is the web framework; no module imports Starlette directly.
- ADC on Agent Platform is the default authentication; API keys are opt-in.
- Every agent speaks through a typed `output_schema`, records its verdict under
  an `output_key` in session state, and every human-input node names a
  `response_schema`.

`tests/test_standards.py` checks the first three mechanically.

## Configuration

Everything lives in [`config/devteam.yaml`](config/devteam.yaml); the schema is
[`src/devteam/config.py`](src/devteam/config.py). The environment carries
secrets only.

**Where calls land.** `gcp.project` and `gcp.location` are the Agent Platform
coordinates for models, sessions, and Memory Bank; the project falls back to
`$GOOGLE_CLOUD_PROJECT`. ADC supplies the identity.

**Models.** A provider entry is `{provider, backend, model, location, generation, extra}`.
`backend` defaults to `agent-platform`: Gemini runs through ADK's `Gemini`
class on Vertex, Claude through ADK's `Claude` class over `AsyncAnthropicVertex`,
both with ADC. `generation` holds typed settings that reach the agent as ADK's
`GenerateContentConfig` (`thinking_level` for Gemini, `effort` for Claude,
`temperature`, `max_output_tokens`). `location` overrides the region for one
model, since Claude is served from fewer regions than Gemini:

```yaml
claude:
  provider: anthropic
  model: claude-opus-5        # current-generation Claude uses bare ids on Agent Platform
  location: us-east5
  generation: {effort: high}
```

`backend: api-key` is the opt-in path: the vendor's own API through LiteLLM,
with `extra` passed verbatim (`thinking`, `api_base`, ...):

```yaml
claude-api:
  provider: anthropic
  backend: api-key
  model: claude-opus-5
  extra: {thinking: {type: adaptive}}
```

**Per-phase loop-spec models.** `loop_spec.phases` and `loop_spec.roles` name
provider entries and are exported as loop-spec's documented
`LOOP_SPEC_PHASE_MODEL_<PHASE>` / `LOOP_SPEC_MODEL_<ROLE>` variables as ADK
registry ids (a bare Gemini id, or `Claude:projects/.../models/<id>` for
Claude on Vertex). Any Agent Platform route also exports
`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`,
because loop-spec builds its role agents from bare model strings:

```yaml
loop_spec:
  phases: {spec: claude, plan: claude, execute: gemini-pro}
  roles: {code_reviewer: claude}
```

**Sessions and memory.** `services.backend: in-memory` runs everything locally;
`agent-platform` uses `VertexAiSessionService` + `VertexAiMemoryBankService`
against `services.agent_platform.agent_engine_id` in the `gcp` project. The
`memory_commit` plugin sends each turn's new events to whichever memory store
is active (so Memory Bank extracts from fresh material, never the whole session
again), and the qa agent reads it back with `preload_memory` / `load_memory`.

**Structured outputs.** Intake returns `IntakeResult {category, request, team}`
and writes it to session state under `intake_verdict`; the router reads the
input first and the stored verdict second. The qa agent keeps its memory and
peer tools while returning `QaAnswer {answer, sources, teams_consulted}` under
`qa_answer` (ADK exposes the tools during the thought loop and enforces the
schema on the final reply). The clarify node asks for a typed
`Clarification {category, request}` and feeds it straight back into the
router. The manager returns `PhaseVerdict {decision, guidance}`, and
loop-spec's result file is read into a typed `CycleResult`.

**A2A teams.** Deploy one instance per repository and point them at each
other. `devteam serve` exposes the whole graph on FastAPI — intake, the manager
loop with its loop-spec engineer, and Q&A — with an agent card at
`/.well-known/agent-card.json` and a health check at `/healthz`, so a feature
filed at any instance is shipped by the instance that owns the repository:
intake names the owning team, the graph's `PEER:<name>` route forwards the
request whole over A2A, and the peer's own intake and engineer take it from
there. The qa agent additionally gets one tool per peer for questions. Because
the surface includes an unsandboxed shell, every request except the card and
the health check must carry `Authorization: Bearer $DEVTEAM_A2A_TOKEN`, and
binding a non-loopback host without a token is refused. `expose.tls` serves
HTTPS directly between containers; a peer's `ca_bundle` names the private CA
that signed its certificate:

```yaml
a2a:
  expose:
    host: 0.0.0.0
    port: 8001
    token_env: DEVTEAM_A2A_TOKEN
    tls: {certfile: /etc/devteam/tls/cert.pem, keyfile: /etc/devteam/tls/key.pem}
  peers:
    - name: platform_team
      url: https://platform-devteam.internal:8001
      description: Owns the platform service and its deployment pipeline.
      token_env: PLATFORM_TEAM_TOKEN
      ca_bundle: /etc/devteam/tls/internal-ca.pem
```

Peers are consumed with `use_legacy=False`, which activates ADK's A2A
extension (no duplicated messages or lost nested output in streaming). The
test suite runs this round trip for real on localhost over both HTTP and
HTTPS with a self-signed certificate.

## The manager loop

A long-horizon run tends to asymptote: the agent gets far, then stalls in the
minutiae. The remedy is a manager that hands the implementer one phase at a
time, checks the checklist between phases, and moves the cycle on when no box
has been ticked for a while. `src/devteam/manager.py` is that pattern as one
native ADK `Workflow` graph over loop-spec's own phases:

```
START ─▶ brief ─▶ implementer ─▶ observe ─┬─▶ JUDGE ─▶ manager ─▶ route_round ─┬─▶ CONTINUE ─▶ implementer
                       ▲                  │                                    ├─▶ ASK ─▶ ask_human ─▶ resume_or_stop ─┬─▶ CONTINUE
                       └──────────────────┴────────────────────────────────────┘                                        └─▶ DONE ─▶ finish
                                          └─▶ DONE ─────────────────────────────────────────────────────────────────────────────▶ finish
```

- **implementer** is the mounted loop-spec working agent. As a workflow node it
  runs single-turn with no prior contents, so every round is a fresh context.
  loop-spec's `supervised` profile makes it return after each durable phase
  with a paused `phase-handoff` result.
- **observe** is deterministic code: it reads `.loop-spec/last-result.json`
  and the PLAN checklist (`features/<slug>/tasks.json`), appends a tick count
  to the progress ledger in session state, and flags a stall (no new tick for
  `stall_rounds` rounds). A terminal result ends the loop here.
- **manager** is an LlmAgent with a typed verdict: `CONTINUE`, `MOVE_ON`, or
  `HALT`, plus one paragraph of guidance the next hand-off carries.
- **route_round** is code again: a halt or a spent `max_rounds` budget asks a
  human (a typed `HumanDecision {resume, guidance}`); anything else re-issues
  the cycle with the guidance and `phase_prompt` appended.

`phase_prompt` asks for each phase "extremely well", not "perfectly": the
former lets the implementer close a phase once it is good enough, the latter
sends it back into the minutiae. `devteam serve` renders the ledger at
`/progress/<user>/<session>` as a checklist page with a counter and ticks over
time. Everything is tuned under `loop_spec.manager` in the YAML.

## The loop-spec mount

loop-spec lives at `third_party/loop-spec` (git submodule) and is imported
directly by the app — features and bugs route into its working agent, which
runs the SPEC → DISCUSS → PLAN → EXECUTE → VERIFY → ITERATE → DELIVER cycle
and ships one verified PR. `devteam supervise` runs the manager loop
unattended on loop-spec's four supervisor ports, each configured under
`loop_spec.supervisor` and `loop_spec.manager`:

| Port | Where it lands |
|---|---|
| decision oracle | `oracle.halt_when` / `oracle.prefer` / `oracle.pins` answer `get_user_choice`; the default is loop-spec's own `(Recommended)` option |
| state store | `store_dir` names loop-spec's `store-mirror.sh` adapter so an ephemeral container survives death |
| event sink | `events_file` names loop-spec's `append-sink.sh` adapter; every cycle event line lands there |
| lifecycle | the manager loop re-issues the cycle after each phase hand-off in a fresh context, an interrupted invocation is resumed by id, and a run that dies before writing its result is reconciled with `cycle-reconcile.sh` |

Success is judged by `outcome` + `converged`, never `status` alone.

loop-spec's EXECUTE fleet rung launches `adk run` against a mounted agent
directory. Generate that machine-local mount once (it is gitignored) and
`devteam` exports `LOOP_SPEC_ADK_AGENT_DIR` whenever it exists:

```sh
bash scripts/mount-loop-spec.sh
uv run adk run adk_agents/loop_spec     # or adk web
```

## Not used, and why

- **ADK Agent Config (YAML agents)** is Gemini-only and experimental; this app
  drives Gemini and Claude alike, so agents stay in typed Python and only
  their settings live in YAML.
- **ADK's `get_fast_api_app`** serves an agents directory with its own REST
  and dev UI; this app is built from one config file, so it mounts the same
  a2a-sdk routes on a lean FastAPI app instead.

## Layout

```
config/devteam.yaml     the one config file
src/devteam/
  config.py             typed YAML schema + loader
  models.py             spec → native ADK model on Agent Platform (ADC) or LiteLLM (api-key)
  services.py           session + memory service pair
  agents.py             intake classifier, qa agent, authenticated A2A peers (typed outputs)
  graph.py              the request Workflow graph (routes, peer forwarding, typed human input)
  manager.py            the manager loop over loop-spec phases, ledger, stall rule, progress page
  cycle.py              loop-spec's result and checklist files as typed records
  app.py                composition root: Apps, Runner, plugins, resumability
  runtime.py            one turn with get_user_choice answered by a YAML oracle policy
  a2a.py                FastAPI A2A server: full graph, bearer auth, optional TLS, /healthz, /progress
  loopspec.py           loop-spec mount + LOOP_SPEC_* and Vertex environment
  supervisor.py         unattended manager-loop runs on the four ports
  cli.py                devteam check | chat | serve | supervise
tests/                  offline suite: scripted LLMs through the real graphs,
                        HTTP + HTTPS localhost A2A round trips incl. peer forwarding,
                        the manager loop against a fake loop-spec, the code standards
third_party/loop-spec   the loop-spec checkout (submodule)
```
