# Ravana — Implementation Task List

Detailed task breakdown for each phase in [ARCHITECTURE.md §12](ARCHITECTURE.md). §12 is the *why this order* (architectural sequencing); this file is the *what to actually build*. Section references (§x.y) point back to ARCHITECTURE.md.

Phase 0 is broken down to the task level since it's what's actionable now — and further split into **0a/0b**, because "MVP" originally bundled a full vertical slice (schema, structured-output enforcement, LLM fallback, MCP, Docker sandbox, git isolation, HITL, audit log, mock mode, E2E) into one 4–6 week estimate, which isn't an MVP-sized unit of work, it's most of the platform. Splitting it separates the part that's genuinely hard and novel (the graph engine's routing/HITL/concurrency logic — already the source of two real bugs found just from writing docs and examples) from the part that's comparatively mechanical integration work (real LLM calls, real tool execution, real git). Phases 1–3 are intentionally less granular — per §12's own reasoning, each phase should surface what the next one actually needs, so committing to a detailed Phase 2 task list before Phase 0 ships would mostly be guessing.

---

## Phase 0a — Deterministic Core (mock agents, no real LLM/sandbox/git)

Goal: prove the graph engine — routing, HITL, loops, concurrency, guards — is correct against **scripted mock agent responses**, before spending any effort on real LLM/tool integration. This is where bugs like the `qa_test` dead-end and the HITL-resume semantics fix are cheap to catch; they'd be expensive and confusing to debug for the first time through a flaky real LLM call.

### Data layer
- [ ] Translate §2.2's Postgres DDL to SQLite (JSONB → JSON columns, UUID → TEXT, `UUID[]`/`TEXT[]` → JSON arrays) — all 12 tables (`agent`, `toolkit`, `skill`, `workflow`, `workflow_node`, `workflow_edge`, `run`, `node_execution`, `message`, `artifact`, `state_transition_log`, `hitl_request`, `audit_log`), including the `workflow_node_backing_xor` CHECK and `workflow_edge`'s `source_node_id` FK / non-empty-targets CHECK
- [ ] `ravana init`: scaffold `.ravana/` (`config.yaml`, `workflows/`, `state.db`, `runs/`, `.gitignore`) per §10.1's directory layout
- [ ] Bootstrap script for `state.db` schema creation (no migration history needed yet — single-user, single-version)

### Workflow compiler
- [ ] Pydantic models for the full YAML schema (§4): `state.schema` (with `merge`/`pii` per key), `toolkits`, `skills`, `agents` (with `llm`/`llm.fallback`/`output_schema`/`hitl`), `graph.nodes`/`graph.edges` (with `is_default`), `guards`, `concurrency`, `definition_of_done`
- [ ] YAML → internal graph compiler (resolve node/edge/agent/toolkit/skill references by id)
- [ ] `ravana workflow validate`: entry-node exists, no unreachable nodes, merge-policy conflicts on broadcast branches (§3.5), non-terminal nodes missing both `is_default` and `hitl_config` (§3.3) flagged as a warning, edge targets resolve to a real node id or `__terminal__` (the DB layer deliberately can't check this, §2.2 — the compiler is where it's actually enforced)

### Orchestrator (single-process — no lease/CAS needed at this tier, §10.1)
- [ ] Engine loop per the corrected §3.1 sequence: dispatch → agent turn → commit (state_version CAS + `state_transition_log` sequence/event_type/node_execution_id, §2.2) → **route-or-pause-or-fail** (conditional edges → HITL check → default edge → fail-fast dead-end) → **resume as a new `node_execution` attempt** (not a bare re-route — §3.1's corrected semantics) → terminate
- [ ] `guards` enforcement: `max_total_steps`, `max_loop_iterations` (per-edge), `max_tool_calls_per_turn`, `max_output_repairs`, `max_retries_per_node`, `max_tokens_total`
- [ ] `workflow.concurrency` group check at Trigger (§3.7) — even single-process, still needs to queue a second local run against the same `repository` group
- [ ] Keep `state_version` bookkeeping even though nothing contends for it yet — keeps the schema forward-compatible with Phase 1's real CAS path

### Mock Agent Runtime
- [ ] A pluggable "mock" backend: instead of calling a real LLM, returns a pre-scripted `structured_payload` per node per turn from a test fixture — this is the primary way 0a gets built and tested, not a testing afterthought bolted on at the end
- [ ] Fixture format that can script a full run including HITL-triggering responses (e.g. "on turn 1, PM emits `requirement_clarity: LOW`; after human response, PM emits `HIGH`") so the corrected Resume mechanic (§3.1) is exercisable without a real model

### HITL
- [ ] `hitl_request` create/resolve, `assignee` field populated even though it's just "the local user" at this tier
- [ ] Blocking terminal prompt in `ravana run watch`

### CLI
- [ ] `ravana init`
- [ ] `ravana workflow validate`
- [ ] `ravana run start`
- [ ] `ravana run watch` (tails `message`/`state_transition_log`, prompts on HITL)
- [ ] `ravana run hitl respond`

### Observability (minimal, per §9)
- [ ] `audit_log` writes on every `DRAFT` save / `publish` / manual action, from day one
- [ ] Structured JSON log lines to stdout/file, tagged with `run_id`/`node_execution_id` (full Loki + correlation dashboards are Phase 2, but the tagging convention should exist from the start so nothing needs retrofitting)

### Testing
- [ ] Unit tests for the routing fix specifically: condition coverage, `is_default` firing, HITL-before-fail-fast ordering, the exact dead-end case (`qa_status == FAIL && iteration_count >= 5`) found in §4's own example
- [ ] Unit tests for the Resume fix: a HITL response produces a *new* `node_execution` attempt for the same node, not just a re-route of stale output
- [ ] Unit test for the corrected idempotency key (§3.6): same tool name + same arguments across two attempts ⇒ same key; different arguments ⇒ different key
- [ ] End-to-end: run the full SDLC example workflow with the mock backend from `pm_intake` to `COMPLETE`, including at least one HITL round-trip and one loop iteration

---

## Phase 0b — Real Integration (LLM, tools, sandbox, git)

Goal: swap the mock backend for real providers and real tool execution, on a graph engine already proven correct in 0a — this phase is comparatively mechanical precisely because 0a already absorbed the hard, novel, bug-prone part.

### Agent Runtime / LLM Gateway
- [ ] Provider adapters: Anthropic (native tool-calling) + one local model via Ollama's OpenAI-compatible endpoint (guided decoding if available, else repair-loop) — per §3.4's capability-ranked strategy selection
- [ ] `submit_result` synthetic tool wiring; within-node tool-use loop bounded by `max_tool_calls_per_turn`, force-terminated via `tool_choice` if the budget runs out
- [ ] `agent.llm_fallback` chain (§3.6): on primary exhaustion, try each fallback entry with its own small retry budget
- [ ] Prompt Assembler: system prompt + injected `shared_state` + short-term memory + Skills (§1.6, always-on concatenation — no progressive disclosure yet)

### Toolkits
- [ ] Toolkit executor interface: JSON Schema in/out; idempotency key computed per call as `hash(run_id, node_id, tool_name, canonical_json(arguments))` (§3.6 — content-addressed, stable across retries by construction, *not* derived from `node_execution.attempt`)
- [ ] Built-in types: `web_search`, `code_interpreter` (local Docker, mounts **only** `runs/<run_id>/workspace`, §10.1 point 4 — never the host, never the parent project directory), `api_connector` (generic HTTP + top-level `auth_ref` resolution), `mcp_server` (stdio transport, official `mcp` Python SDK, §1.7)

### Git isolation (§10.1 — the "don't touch source" requirement)
- [ ] `git clone --local` into `runs/<run_id>/workspace` on branch `ravana/run-<run_id>` (default)
- [ ] `git worktree add` as an opt-in alternative
- [ ] `git init` shadow-repo fallback when the target isn't a git repo
- [ ] On `COMPLETE`: `git_connector` opens a PR (or writes a patch file if no remote) — never auto-merges

### CLI / UX
- [ ] `ravana studio` (localhost dev-server UI reading `state.db` directly — can slip to Phase 1 if it doesn't fit the timebox)

### Testing
- [ ] Replace the SDLC E2E's mock fixtures with real-LLM smoke tests, one per provider adapter
- [ ] Verify structured-output strategy selection (§3.4) actually resolves to the expected mechanism per provider (guided decoding for the local model, native tool-calling for Anthropic)
- [ ] Idempotency integration test: force a retry against a fake connector and confirm the *second* call is recognized as a duplicate, not just that the key looks stable in isolation

---

## Phase 1 — Self-hosted (single instance) & UI

Goal: `docker compose up` gives a team the Ravana Console with all three UI surfaces (§1.5) and real concurrency.

- [ ] **Data layer**: Postgres DDL from §2.2 (the real thing, not the SQLite translation); Alembic migrations from here on; one-time import path from a Phase-0 `.ravana/state.db` so early local runs aren't stranded
- [ ] **Orchestrator scaling**: `node_execution` lease claiming (`UPDATE ... WHERE status='QUEUED'` / `FOR UPDATE SKIP LOCKED`); `state_version` CAS commit with merge-policy-aware retry (§3.5) now actually exercised; broadcast-edge parallel dispatch; Redis Streams as the lease/HITL transport
- [ ] **Control-plane API**: FastAPI implementing the full §7 contract (agents/toolkits/workflows/runs/hitl/audit-log), SSE `/runs/{id}/stream`, `DRAFT`→`PUBLISHED` lifecycle + `/publish`, full `/validate`
- [ ] **Ravana Console**: React Flow graph editor (Design), generated intake form (Runs → New Run), operator/monitoring view with live stream + HITL respond (Runs → [a run]), version-history tab off `/audit-log`
- [ ] **Sandbox**: `code_interpreter` optionally routes to a managed provider (E2B/Modal) — config-driven per Toolkit, not yet the default
- [ ] **Connector SDK**: documented + contract-test harness, opened to third-party toolkit authors
- [ ] **HITL**: Slack notification integration
- [ ] **Observability**: OpenTelemetry spans per `node_execution` attempt begin here (full dashboards can wait for Phase 2's Grafana setup)
- [ ] **Testing**: docker-compose based integration environment for CI

---

## Phase 2 — Production hardening (scaled self-hosted)

Goal: multi-team, multi-domain org running Ravana on Kubernetes with real tenancy.

- [ ] Kubernetes Helm chart (API / orchestrator / worker pool / sandbox pool as independently-scaled deployments, per §10.3's diagram)
- [ ] `org_id` enforcement across every query and API auth check (multi-tenancy stops being theoretical)
- [ ] RBAC roles (Workflow Author / Operator / Viewer) and auth middleware
- [ ] Vault/KMS integration replacing any placeholder secret resolution
- [ ] Managed sandbox becomes the default backend (not opt-in) — multi-tenant workflows are now the norm, per §8
- [ ] Golden-run regression suite + eval harness gating `publish` (§11)
- [ ] pgvector long-term memory (§1.3)
- [ ] Workflow composition / sub-workflows (§1.8): `workflow_node.sub_workflow_id`, nested-run dispatch, `output_map` commit — build only once a real cross-workflow reuse need shows up
- [ ] Per-org backpressure **and** per-credential rate limiting (§9) — the two are different failure modes, both needed
- [ ] Loki + mandatory `run_id`/`node_execution_id` correlation across infra logs (§9)
- [ ] Stuck-run detection/alerting on `WAITING_HUMAN` timeout and heartbeat-less `RUNNING`

---

## Phase 3 — Scale-out / Managed Cloud

Goal: multi-region throughput, and a Managed Cloud tier that's an actual business, not just Phase 2's architecture with a login page.

- [ ] Kafka event bus, replacing Redis Streams for multi-region throughput
- [ ] Supervisor/dynamic LLM routing (§3.3) — an LLM-chosen `next_node` for topologies that don't reduce to boolean conditions
- [ ] Community Toolkit marketplace (submission/review process, registry, versioning)
- [ ] Data governance: enforce the `pii: true` state-key flag end-to-end, build a right-to-delete flow that redacts flagged content while preserving non-PII audit rows
- [ ] Billing/usage metering: roll up `node_execution.estimated_cost_usd` per `org_id` per billing period into a metered-billing integration (e.g. Stripe)
- [ ] SOC2 compliance program (operational, not architectural — start the clock early since certification timelines are long)
- [ ] Multi-region deployment topology
