"""The LLM Gateway (§1.1, §3.4) — the real `AgentRuntime` for Phase 0b, a
drop-in replacement for the Phase-0a MockAgentRuntime behind the same
`run_turn` protocol, so the engine loop is unchanged.

Per node dispatch it:
1. Assembles the prompt (persona + skills + state) — ravana.runtime.prompt.
2. Picks the strongest structured-output strategy the agent's provider
   supports (§3.4): guided decoding > native/forced-tool > repair-loop.
3. Runs the within-turn tool-use loop: LLM -> (tool call -> execute -> feed
   back)* -> `submit_result`. Bounded by guards.max_tool_calls_per_turn; the
   turn ends specifically on `submit_result` (not on "the model went quiet").
4. On invalid `submit_result` args, re-prompts with the validation error up
   to guards.max_output_repairs (§3.4 tier 3), then gives up.
5. Wraps the whole thing in the llm_fallback chain (§3.6): primary model,
   then each fallback entry with its own small budget.

Tool *execution* (the "-> execute ->" step) is delegated to an injected
ToolExecutor. Phase 0b's LLM-gateway slice ships the loop and a no-op
executor; real toolkits (code_interpreter, mcp_server, ...) are the next
slice and slot in behind this same protocol.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ravana.compiler.graph import CompiledGraph, NodeExecutionContract
from ravana.runtime.backoff import RetrySleep, backoff_delay
from ravana.runtime.base import AgentOutputError, AgentTurnResult, ProseJudgement, TransientAgentError
from ravana.runtime.idempotency import compute_idempotency_key
from ravana.runtime.prompt import assemble_system_prompt
from ravana.runtime.schema_validate import validate_json
from ravana.runtime.secrets import ResolvedSecret, SecretResolver, ensure_secret_free
from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.runtime.providers.base import (
    AssistantMessage,
    Capability,
    Message,
    ProviderAdapter,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderTarget,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from ravana.schema.models import LLMConfig, LLMFallbackEntry

SUBMIT_RESULT = "submit_result"
# §3.1 step 7: the forced tool a non-guided provider submits its prose DoD
# verdict through (the guided path returns the same shape as JSON text).
SUBMIT_VERDICT = "submit_verdict"

# §3.6: per-fallback-entry retry budget — deliberately small (1) and separate
# from the engine-level guards.max_retries_per_node, so a long fallback chain
# can't multiply total attempts.
_PER_ENTRY_RETRIES = 1

# §3.6 backoff shape for a same-entry retry (a 429/5xx wants breathing room).
# Smaller cap than the engine's per-node backoff: this is the inner loop — the
# engine's own retry adds its own, larger delays on top.
_ENTRY_RETRY_BASE_SECONDS = 1.0
_ENTRY_RETRY_CAP_SECONDS = 10.0

# A schema-less fallback when a node declares no output_schema: submit_result
# then accepts any object, and the whole returned object becomes the
# state_delta. (A stricter default could require declared state keys; kept
# permissive here so agents without output_schema still work.)
_ANY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}

# §3.1 step 7: the structured shape the evaluated_by agent returns for a prose
# DoD judgement — one {index, met} per criterion. Indices (not the criterion
# text) key each verdict, so a long free-text criterion never has to survive a
# round-trip as a JSON key. `reason` is optional: it only nudges the model to
# justify its call and is NEVER persisted — the DoD event logs the
# workflow-authored criteria plus the booleans, no model free-text (§8).
_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "met": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "met"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

_JUDGE_INSTRUCTION = (
    "You are now acting as the Definition-of-Done judge for a completed run. "
    "For each prose criterion you are given, decide whether the shared state "
    "above is sufficient evidence that it is met. Judge conservatively: if the "
    "evidence does not clearly establish a criterion, mark it NOT met."
)


class ToolExecutor(Protocol):
    """Executes a real (non-submit_result) tool call and returns a result
    string to feed back into the turn, and describes an agent's toolkits as
    callable-tool specs the gateway surfaces to the model.

    `idempotency_key` identifies a logical tool invocation (§3.6), computed by
    the gateway and passed in **before** execution — a side-effecting connector
    (git push, ticket create) MUST dedupe on it, which is only possible if it
    has the key at execution time, not after. The engine's later persistence
    of the same key into message.tool_calls (loop.py) uses this same value."""

    def tools_for(self, toolkit_ids: list[str]) -> list[Tool]: ...

    async def execute(
        self, *, run_id: str, node_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str
    ) -> str: ...

    async def aclose(self) -> None: ...


class _NoToolExecutor:
    """The default when no real executor is injected (the gateway's own test
    fakes, or a run with no toolkits): it surfaces no tools, so the turn is
    submit_result-only, and any tool call is a hard error."""

    def tools_for(self, toolkit_ids: list[str]) -> list[Tool]:
        return []

    async def execute(
        self, *, run_id: str, node_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str
    ) -> str:
        # A wiring bug, not a provider fault — deliberately NOT ProviderError,
        # which the fallback loop would retry/fall through (each pass burning
        # full LLM turns), nor ToolkitError, which the tool loop would feed
        # back to the model to try again. No executor means no tool can ever
        # run; fail the run hard via the engine's terminal boundary.
        raise RuntimeError(
            f"agent tried to call tool '{tool}' but no ToolExecutor is wired "
            "(no toolkits are available to this run)"
        )

    async def aclose(self) -> None:
        return None


@dataclass
class _Strategy:
    """The resolved structured-output approach for one agent (§3.4), decided
    once from the provider's capabilities rather than per call."""

    use_guided: bool
    use_native_forced_tool: bool
    # If neither of the above, the only lever left is the repair loop.


def _select_strategy(adapter: ProviderAdapter, target: ProviderTarget) -> _Strategy:
    caps = adapter.capabilities(target)
    if Capability.GUIDED_DECODING in caps:
        return _Strategy(use_guided=True, use_native_forced_tool=False)
    if Capability.NATIVE_STRUCTURED_OUTPUT in caps:
        return _Strategy(use_guided=False, use_native_forced_tool=True)
    return _Strategy(use_guided=False, use_native_forced_tool=False)


def _validate(payload: Any, output_schema: dict[str, Any] | None) -> str | None:
    return validate_json(payload, output_schema)


def _parse_json(text: str | None) -> tuple[Any, str | None]:
    """(payload, error) — parses the guided model's JSON response text.
    Guided decoding should make this always succeed, but a nominal/hosted
    guided path can still return malformed JSON, so failures feed the repair
    loop rather than crashing."""
    if not text:
        return None, "guided model returned empty output"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"output was not valid JSON: {exc}"


def _ensure_provider_response_secret_free(
    response: ProviderResponse, api_key: ResolvedSecret | None
) -> None:
    values = (api_key.value(),) if api_key is not None else ()
    ensure_secret_free(
        {
            "text": response.text,
            "tool_calls": [
                {"id": call.id, "tool": call.tool, "arguments": call.arguments}
                for call in response.tool_calls
            ],
            "stop_reason": response.stop_reason,
        },
        context="provider response",
        values=values,
    )


class LLMGateway:
    def __init__(
        self,
        graph: CompiledGraph,
        adapters: dict[str, ProviderAdapter],
        tool_executor: ToolExecutor | None = None,
        retry_sleep: RetrySleep = asyncio.sleep,
        secret_resolver: SecretResolver | None = None,
    ):
        self._graph = graph
        self._adapters = adapters
        self._tools = tool_executor or _NoToolExecutor()
        # §3.6 backoff waiter for same-entry retries; injectable so tests
        # record requested delays instead of actually waiting.
        self._retry_sleep = retry_sleep
        # §8c: resolves `llm.api_key_ref` pointers to real keys at dispatch —
        # the gateway (the Agent Runtime layer) does the resolving; adapters
        # only ever receive the resolved value. None is fine for workflows
        # whose agents declare no api_key_ref (SDK env vars serve them).
        # Deliberately NO key memo here: resolution is per entry-try (see
        # _resolve_api_key) so there is no shared mutable credential state.
        self._secret_resolver = secret_resolver
        # §3.4: strategy is decided once per immutable provider target, not
        # re-derived on every call.
        self._strategy_cache: dict[tuple[int, ProviderTarget], _Strategy] = {}

    async def aclose(self) -> None:
        """Close every execution-plane resource owned by this gateway."""
        first_error: RuntimeError | None = None
        resources = [*self._adapters.values(), self._tools]
        seen: set[int] = set()
        for resource in resources:
            identity = id(resource)
            if identity in seen:
                continue
            seen.add(identity)
            close = getattr(resource, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:  # noqa: BLE001 - continue closing siblings
                first_error = first_error or RuntimeError(
                    f"runtime resource cleanup failed ({type(exc).__name__})"
                )
        if first_error is not None:
            raise first_error

    def _adapter_for(self, provider: str) -> ProviderAdapter:
        adapter = self._adapters.get(provider)
        if adapter is None:
            # Config error, and permanent BY ENTRY: retrying this entry can't
            # conjure the adapter, but the next fallback entry may name a
            # provider that IS registered — so it stays a ProviderError
            # (chain moves on), just never same-entry retried.
            raise ProviderError(f"no adapter registered for provider '{provider}'", retryable=False)
        return adapter

    def _resolve_api_key(self, llm: LLMConfig) -> ResolvedSecret | None:
        """§8c: turn the entry's `api_key_ref` pointer into the real key.
        Resolved FRESH once per entry-try (called at the top of _run_one_llm)
        with NO gateway-level memo: "resolved at dispatch time" means each
        dispatch re-reads the secret — a rotated credential is picked up on
        the next turn — and, with no shared mutable state, concurrent
        run_turn()s cannot swap each other's credentials mid-turn (review
        probe: key-1/key-2 interleaving under a shared per-dispatch dict).
        A resolution failure is a config error, permanent BY ENTRY (a retry
        can't conjure the secret) — but the next fallback entry may use a
        different ref/provider, so it raises ProviderError (chain moves on)
        rather than a hard error.

        Error messages are deliberately FIXED-SHAPE: they carry the ref (a
        schema-enforced `secrets://` pointer, safe to echo) and the exception
        TYPE — never the exception text, which a misbehaving resolver could
        fill with the secret itself. These messages end up in
        node_execution.error and stderr logs (§8: secrets never persisted)."""
        ref = llm.api_key_ref
        if ref is None:
            return None  # no per-agent key: the SDK's own env var serves this entry
        if self._secret_resolver is None:
            raise ProviderError(f"llm.api_key_ref '{ref}' declared but no secret resolver is wired", retryable=False)
        try:
            # SecretResolver.resolve returns ResolvedSecret (non-empty by
            # construction, self-redacting), so a "" resolution can't slip
            # past the adapters' truthiness gates and swap in the SDK's
            # ambient env credential.
            return self._secret_resolver.resolve(ref)
        except Exception as exc:  # noqa: BLE001 - resolution failure is a permanent entry failure
            raise ProviderError(
                f"resolving llm.api_key_ref '{ref}' failed ({type(exc).__name__})", retryable=False
            ) from exc

    def _strategy_for(self, adapter: ProviderAdapter, llm: LLMConfig) -> _Strategy:
        target = ProviderTarget(provider=llm.provider, model=llm.model, endpoint=llm.endpoint)
        key = (id(adapter), target)
        if key not in self._strategy_cache:
            self._strategy_cache[key] = _select_strategy(adapter, target)
        return self._strategy_cache[key]

    async def run_turn(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        logical_visit_id: str,
        agent_id: str,
        shared_state: dict[str, Any],
    ) -> AgentTurnResult:
        agent = self._graph.agents_by_id[agent_id]
        contract = self._graph.contract_for_node(node_id)
        if self._graph.node(node_id).agent != agent_id:
            raise ValueError(f"node '{node_id}' is not backed by agent '{agent_id}'")
        system = assemble_system_prompt(agent, self._graph.skills_by_id, shared_state)
        output_schema = contract.output_schema or _ANY_OBJECT_SCHEMA
        chain = [agent.llm, *(_fallback_to_llm(f) for f in agent.llm.fallback)]

        async def run_one(llm: LLMConfig) -> AgentTurnResult:
            return await self._run_one_llm(
                run_id=run_id, node_id=node_id, logical_visit_id=logical_visit_id,
                contract=contract, llm=llm, system=system, output_schema=output_schema,
            )

        return await self._run_chain(agent_id=agent_id, chain=chain, run_one=run_one)

    async def _run_chain(
        self,
        *,
        agent_id: str,
        chain: list[LLMConfig],
        run_one: Callable[[LLMConfig], Awaitable[AgentTurnResult]],
    ) -> AgentTurnResult:
        """§3.6 fallback chain, shared by node dispatch and DoD prose judgement:
        try the primary llm, then each fallback entry, each with its own small
        retry budget (default 1 retry per entry — distinct from, and smaller
        than, the engine-level max_retries_per_node, so a chain of N fallbacks
        can't multiply total attempts by N). A provider-level failure
        (ProviderError) is what a retry/fallback responds to; an AgentOutputError
        from repair-budget exhaustion is the model producing bad output, not a
        provider fault, so it propagates immediately — past this chain AND past
        the engine's transient retry (§3.6: "repair budget exhausted" is
        non-transient). Only when every entry's budget is spent does the caller
        fail. `run_one` is what differs between callers (a node runs the tool
        loop; a judgement runs a no-tools structured completion)."""
        last_error: Exception | None = None
        # Whether any entry's TERMINAL outcome was retryable. Per-entry, not
        # per-failure: an entry whose first failure was a 500 but whose retry
        # died on a 401 ENDED permanent — a historical transient must not make
        # a hopeless chain look worth the engine's node retries.
        any_entry_ended_retryable = False
        for llm in chain:
            entry_error: ProviderError | None = None
            for try_index in range(_PER_ENTRY_RETRIES + 1):
                try:
                    return await run_one(llm)
                except ProviderError as exc:
                    entry_error = exc
                    last_error = exc
                    # §3.6 taxonomy: a PERMANENT failure (auth 401, invalid
                    # request 400, unknown model) cannot succeed on a same-entry
                    # retry — skip the retry AND its backoff sleep, and move
                    # straight to the next fallback entry (a different
                    # provider/model may well work).
                    if not exc.retryable:
                        break
                    # Retryable (429/5xx/timeout): back off before retrying the
                    # SAME entry — the provider needs breathing room. Moving to
                    # the NEXT entry sleeps zero either way: it's a different
                    # provider, and waiting on it would only delay the recovery
                    # the chain exists to provide.
                    if try_index < _PER_ENTRY_RETRIES:
                        await self._retry_sleep(
                            backoff_delay(try_index + 1, base=_ENTRY_RETRY_BASE_SECONDS, cap=_ENTRY_RETRY_CAP_SECONDS)
                        )
                    continue
            if entry_error is not None and entry_error.retryable:
                any_entry_ended_retryable = True
        if any_entry_ended_retryable:
            # At least one entry ENDED on a transient fault — the engine's own
            # §3.6 retry (fresh attempt, larger backoff) can plausibly recover.
            raise TransientAgentError(f"all LLM entries exhausted for agent '{agent_id}': {last_error}")
        # Every entry ENDED permanently (auth/config/bad request): re-running
        # the chain cannot succeed, so this must NOT look transient — raise a
        # hard error the engine fails the run on immediately, instead of
        # burning max_retries_per_node re-running a hopeless chain.
        raise RuntimeError(
            f"all LLM entries failed permanently for agent '{agent_id}' (auth/config, not transient): {last_error}"
        )

    async def _run_one_llm(
        self,
        *,
        run_id: str,
        node_id: str,
        logical_visit_id: str,
        contract: NodeExecutionContract,
        llm: LLMConfig,
        system: str,
        output_schema: dict[str, Any],
    ) -> AgentTurnResult:
        adapter = self._adapter_for(llm.provider)
        strategy = self._strategy_for(adapter, llm)
        # §8c: resolve THIS entry's key once per try, as a local — the value
        # flows down the call stack (no gateway-level shared credential state,
        # so concurrent run_turn()s can never swap each other's keys mid-turn).
        api_key = self._resolve_api_key(llm)
        # Guided decoding constrains the *entire* response to the output schema,
        # so the payload arrives as JSON in the message text (not a
        # submit_result tool call) and it's one-shot by construction — but that
        # also means the model can't emit tool calls under it. So the guided
        # one-shot path is only usable when the agent has NO toolkits; an agent
        # WITH toolkits must run the tool loop (submit_result forces structure
        # at the end), even on a guided-capable provider. Otherwise a guided
        # agent would silently never see its tools.
        if strategy.use_guided and not contract.toolkits:
            return await self._run_guided(
                node_id=node_id, llm=llm, adapter=adapter, system=system, output_schema=output_schema, api_key=api_key
            )
        return await self._run_tool_loop(
            run_id=run_id, node_id=node_id, logical_visit_id=logical_visit_id,
            contract=contract, llm=llm, adapter=adapter, system=system,
            output_schema=output_schema, api_key=api_key,
        )

    def _request_for(
        self,
        llm: LLMConfig,
        *,
        api_key: ResolvedSecret | None,
        system: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        force_tool: str | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> ProviderRequest:
        """One place that turns an LLMConfig entry into a ProviderRequest —
        the llm-derived fields (model/sampling/endpoint) travel together with
        the per-try resolved key; the per-call shape is the three explicit,
        type-checked options: `tools`+`force_tool` (the tool loop) or
        `output_schema` (guided one-shot)."""
        return ProviderRequest(
            model=llm.model, system=system, messages=messages,
            temperature=llm.temperature, max_tokens=llm.max_tokens,
            endpoint=llm.endpoint, api_key=api_key,
            tools=tools or [], force_tool=force_tool, output_schema=output_schema,
        )

    async def _run_guided(self, *, node_id, llm, adapter, system, output_schema, api_key):
        guards = self._graph.doc.spec.graph.guards
        messages: list[Message] = [UserMessage(text="Respond with your final structured output as JSON.")]
        input_tokens = output_tokens = repair_count = 0
        while True:
            request = self._request_for(llm, api_key=api_key, system=system, messages=messages, output_schema=output_schema)
            response = await adapter.complete(request)
            _ensure_provider_response_secret_free(response, api_key)
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            payload, error = _parse_json(response.text)
            if error is None:
                error = _validate(payload, output_schema)
            if error is None:
                return AgentTurnResult(
                    structured_payload=payload, content=response.text,
                    input_tokens=input_tokens, output_tokens=output_tokens, repair_count=repair_count,
                )
            if repair_count >= guards.max_output_repairs:
                # §3.6: repair exhaustion is NON-transient — the run fails now;
                # a node retry would just re-ask past an expired budget.
                raise AgentOutputError(f"node '{node_id}' guided output invalid after {repair_count} repairs: {error}")
            repair_count += 1
            messages.append(UserMessage(text=f"That output was invalid: {error}. Return valid JSON matching the schema."))

    async def _run_tool_loop(
        self,
        *,
        run_id,
        node_id,
        logical_visit_id,
        contract,
        llm,
        adapter,
        system,
        output_schema,
        api_key,
    ):
        guards = self._graph.doc.spec.graph.guards
        # §3.4.4: the node's resolved tool grants are offered as callable tools
        # (name = toolkit id), plus the synthetic submit_result the turn
        # terminates on. A toolkit named submit_result would shadow that
        # terminator, so reject the collision loudly.
        agent_tools = self._tools.tools_for(list(contract.toolkits))
        if any(t.name == SUBMIT_RESULT for t in agent_tools):
            # A config bug, not a transient fault — raise something the fallback
            # loop (which only retries ProviderError) won't mask as retryable,
            # so it surfaces immediately. A compile-time reserved-id check is a
            # cleaner future home for this.
            raise ValueError(f"toolkit id '{SUBMIT_RESULT}' collides with the reserved submit tool")
        submit_tool = Tool(
            name=SUBMIT_RESULT,
            description="Call this exactly once, when done, to submit your final structured result.",
            input_schema=output_schema,
        )
        offered_tools = [*agent_tools, submit_tool]
        # The set the model is actually permitted to *execute* this turn (not
        # submit_result, which terminates the turn and is handled separately).
        allowed_tool_names = {t.name for t in agent_tools}
        messages: list[Message] = [
            UserMessage(text="Complete your task, then call submit_result with your structured output.")
        ]
        tool_call_count = repair_count = input_tokens = output_tokens = 0
        recorded_tool_calls: list[dict[str, Any]] = []

        # Absolute ceiling on model round-trips. Forcing submit_result at budget
        # exhaustion terminates the turn for a *cooperative* provider, but a
        # non-compliant one (a local runtime that ignores tool_choice, say)
        # could keep answering without ever submitting — so bound the loop
        # itself. The ceiling clears every legitimate path: up to
        # max_tool_calls_per_turn tool round-trips, the forced-submit turn, and
        # up to max_output_repairs repairs, plus slack.
        max_model_calls = guards.max_tool_calls_per_turn + guards.max_output_repairs + 2
        model_calls = 0

        while True:
            if model_calls >= max_model_calls:
                # §3.6 "guard exceeded" is non-transient: a provider that
                # ignored the forced submit for a whole turn's ceiling will
                # ignore it again on a retried attempt.
                raise AgentOutputError(
                    f"node '{node_id}' did not produce a valid submit_result within {max_model_calls} model turns "
                    f"(provider ignored the forced submit?)"
                )
            model_calls += 1
            # §3.4.4: submit_result is offered as a tool and the prompt asks the
            # model to call it, but tool_choice is force-set to it ONLY once the
            # tool budget is spent — forcing every turn would drop the model's
            # free-text reasoning (§3.4.2). Forcing at exhaustion still
            # guarantees the turn terminates rather than looping or going quiet.
            force_submit = tool_call_count >= guards.max_tool_calls_per_turn
            request = self._request_for(
                llm, api_key=api_key, system=system, messages=messages,
                tools=offered_tools, force_tool=SUBMIT_RESULT if force_submit else None,
            )
            response = await adapter.complete(request)
            _ensure_provider_response_secret_free(response, api_key)
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens

            submit_calls = [tc for tc in response.tool_calls if tc.tool == SUBMIT_RESULT]
            submit = submit_calls[0] if submit_calls else None
            other_calls = [tc for tc in response.tool_calls if tc.tool != SUBMIT_RESULT]

            # Real tool calls take precedence over a co-occurring submit_result.
            # A model that calls a tool AND submits in the same response hasn't
            # seen the tool's result yet, so its submit is premature — accepting
            # it would drop the real tool call and let the payload claim work
            # that never ran. Execute the tools this turn; defer the submit.
            if other_calls:
                # Preserve the assistant's tool_calls turn before the results —
                # real providers reject a tool result with no preceding
                # assistant tool_calls message. Every tool_use gets a matching
                # tool_result (real or error) so the transcript stays balanced.
                messages.append(AssistantMessage(tool_calls=response.tool_calls, text=response.text))
                for tc in other_calls:
                    tool_call_count += 1
                    # §8 per-node tool boundary (ARCHITECTURE §916): an agent may
                    # only run tools it was granted, even if the provider names
                    # another registered tool. The offered-tools list is a
                    # prompt-level hint; THIS is the enforcement point.
                    if tc.tool not in allowed_tool_names:
                        messages.append(_error_result(tc, f"tool '{tc.tool}' is not available to this agent"))
                        continue
                    # §3.4.4 guard: a single response can carry several tool
                    # calls, so bound side effects WITHIN the batch too — a
                    # multi-call response mustn't fire past max_tool_calls_per_turn
                    # before the next iteration's force-submit kicks in.
                    if tool_call_count > guards.max_tool_calls_per_turn:
                        messages.append(_error_result(tc, "per-turn tool-call budget exhausted — call submit_result now"))
                        continue
                    key = compute_idempotency_key(
                        run_id,
                        node_id,
                        logical_visit_id,
                        tool_call_count,
                        tc.tool,
                        tc.arguments,
                    )
                    try:
                        result = await self._tools.execute(
                            run_id=run_id, node_id=node_id, tool=tc.tool, arguments=tc.arguments, idempotency_key=key
                        )
                    except ToolkitError as exc:
                        # §3.6 routes each ToolFailureKind differently:
                        if exc.kind is ToolFailureKind.FATAL:
                            # Tool auth failure — non-transient, fails the run
                            # (neither the model nor a retry fixes credentials).
                            raise RuntimeError(f"tool '{tc.tool}' failed fatally: {exc}") from exc
                        if exc.kind is ToolFailureKind.TRANSIENT:
                            # Tool timeout / 5xx — transient: end the turn so
                            # the ENGINE retries a fresh node_execution attempt
                            # with backoff. Side effects already fired in this
                            # turn are deduped by logical invocation identity.
                            raise TransientAgentError(f"tool '{tc.tool}' failed transiently: {exc}") from exc
                        # MODEL_ADDRESSABLE (404/422/bad args): feed the error
                        # back so the model can adjust or route around it.
                        messages.append(_error_result(tc, f"tool '{tc.tool}' failed: {exc}"))
                        continue
                    ensure_secret_free(result, context=f"tool '{tc.tool}' result")
                    recorded_tool_calls.append(
                        {
                            "tool": tc.tool,
                            "arguments": tc.arguments,
                            "logical_visit_id": logical_visit_id,
                            "tool_call_ordinal": tool_call_count,
                            "idempotency_key": key,
                        }
                    )
                    messages.append(ToolResultMessage(tool_call_id=tc.id, tool=tc.tool, content=result))
                # EVERY submit_result that rode along (a provider can emit more
                # than one) needs a matching tool_result, or the transcript is
                # unbalanced and a strict provider (OpenAI) rejects the next
                # request. We won't honor a premature submit — feed each an
                # "ignored" result.
                for sc in submit_calls:
                    messages.append(_error_result(sc, "submit_result ignored: you still had pending tool calls; submit again after reviewing their results"))
                continue

            if submit is not None:
                error = _validate(submit.arguments, output_schema)
                if error is None:
                    return AgentTurnResult(
                        structured_payload=submit.arguments,
                        content=response.text,
                        tool_calls=recorded_tool_calls,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        repair_count=repair_count,
                        # Report the count of tool calls actually EXECUTED (not
                        # refused/over-budget attempts), so the engine's post-turn
                        # max_tool_calls_per_turn guard reflects real side effects
                        # and doesn't fail a run the gateway already capped.
                        tool_call_count=len(recorded_tool_calls),
                    )
                if repair_count >= guards.max_output_repairs:
                    # §3.6: repair exhaustion is NON-transient — fail the run
                    # now rather than re-asking past an expired budget.
                    raise AgentOutputError(
                        f"node '{node_id}' output failed validation after {repair_count} repairs: {error}"
                    )
                repair_count += 1
                messages.append(UserMessage(text=f"Your submit_result was invalid: {error}. Try again."))
                continue

            # No tool calls at all — nudge toward submit_result.
            messages.append(UserMessage(text="Call submit_result now with your final output."))
            tool_call_count += 1

    async def judge_prose(
        self, agent_id: str, criteria: list[str], state: dict[str, Any]
    ) -> ProseJudgement:
        """§3.1 step 7: ask the `evaluated_by` agent to rule on prose DoD
        criteria against the final shared state. Returns a ProseJudgement whose
        `verdicts` are **position-aligned** to `criteria` (verdicts[i] rules on
        criteria[i]) plus the LLM usage spent. Runs through the same §3.6
        fallback chain as node dispatch (a judgement should survive a primary
        provider outage the same way a node does), and carries token usage back
        so the engine can meter it against guards.max_tokens_total. This
        method's shape IS the engine's async ProseVerdict."""
        agent = self._graph.agents_by_id.get(agent_id)
        if agent is None:
            # A DoD naming an unknown evaluated_by agent is a config error. Raise
            # (not silently pass) so the DoD gate's fail-closed boundary FAILs
            # the run rather than completing on prose it never judged. (Also
            # rejected at compile time — this is the runtime backstop.)
            raise ValueError(f"definition_of_done.evaluated_by names unknown agent '{agent_id}'")
        system = assemble_system_prompt(agent, self._graph.skills_by_id, state) + "\n\n" + _JUDGE_INSTRUCTION
        numbered = "\n".join(f"{i}: {criterion}" for i, criterion in enumerate(criteria))
        user_text = (
            "Rule on each Definition-of-Done criterion below, using ONLY the "
            "shared state above as evidence. Return a verdict for EVERY "
            f"criterion, keyed by its index.\n\nCriteria:\n{numbered}"
        )
        chain = [agent.llm, *(_fallback_to_llm(f) for f in agent.llm.fallback)]

        async def run_one(llm: LLMConfig) -> AgentTurnResult:
            return await self._judge_completion(llm, system, user_text)

        turn = await self._run_chain(agent_id=agent_id, chain=chain, run_one=run_one)
        return ProseJudgement(
            verdicts=_map_verdicts(turn.structured_payload, len(criteria)),
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
        )

    async def _judge_completion(self, llm: LLMConfig, system: str, user_text: str) -> AgentTurnResult:
        """A no-tools structured completion returning one _VERDICT_SCHEMA object,
        wrapped as an AgentTurnResult so it flows through `_run_chain` (fallback
        + retry) and carries token usage for the engine to meter. Reuses the
        node path's strategy pick (guided vs forced-tool) and small helpers, but
        deliberately offers NO toolkits and skips the tool loop's
        idempotency/bookkeeping — a judgement has no side effects. Bounded by
        guards.max_output_repairs; exhausting that raises AgentOutputError
        (non-transient — the DoD gate turns it into a fail-closed run)."""
        adapter = self._adapter_for(llm.provider)
        strategy = self._strategy_for(adapter, llm)
        api_key = self._resolve_api_key(llm)
        guards = self._graph.doc.spec.graph.guards
        messages: list[Message] = [UserMessage(text=user_text)]
        input_tokens = output_tokens = repair_count = 0
        while True:
            if strategy.use_guided:
                request = self._request_for(
                    llm, api_key=api_key, system=system, messages=messages, output_schema=_VERDICT_SCHEMA
                )
            else:
                verdict_tool = Tool(
                    name=SUBMIT_VERDICT,
                    description="Submit your Definition-of-Done verdict, one entry per criterion.",
                    input_schema=_VERDICT_SCHEMA,
                )
                request = self._request_for(
                    llm, api_key=api_key, system=system, messages=messages,
                    tools=[verdict_tool], force_tool=SUBMIT_VERDICT,
                )
            response = await adapter.complete(request)
            _ensure_provider_response_secret_free(response, api_key)
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            if strategy.use_guided:
                payload, error = _parse_json(response.text)
            else:
                calls = [tc for tc in response.tool_calls if tc.tool == SUBMIT_VERDICT]
                payload, error = (calls[0].arguments, None) if calls else (None, "no submit_verdict call returned")
            if error is None:
                error = _validate(payload, _VERDICT_SCHEMA)
            if error is None:
                return AgentTurnResult(
                    structured_payload=payload, input_tokens=input_tokens,
                    output_tokens=output_tokens, repair_count=repair_count,
                )
            if repair_count >= guards.max_output_repairs:
                raise AgentOutputError(
                    f"DoD prose judgement produced invalid output after {repair_count} repairs: {error}"
                )
            repair_count += 1
            messages.append(
                UserMessage(text=f"That verdict was invalid: {error}. Return valid output matching the schema.")
            )


def _map_verdicts(payload: dict[str, Any], n: int) -> list[bool]:
    """Turn a raw judgement payload into position-aligned booleans for `n` prose
    criteria, FAIL-CLOSED. The shallow schema check only guarantees the
    `verdicts` key exists — not that it's a well-formed list of {index, met} —
    so map defensively:
      - a criterion is met only when its entry's `met` is *exactly* True (a
        truthy non-True value like the string "false" must not read as met);
      - `index` must be a real int in `[0, n)` — a bool is rejected (`bool` is a
        subclass of `int`, so `index: false` would otherwise collide with
        criterion 0), and an out-of-range index is dropped;
      - a duplicate ruling on the same index is unreliable, so that criterion
        fails closed;
      - any criterion with no usable entry is not met.
    """
    met_by_index: dict[int, bool] = {}
    duplicates: set[int] = set()
    verdicts = payload.get("verdicts")
    for item in verdicts if isinstance(verdicts, list) else []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if type(index) is not int or not (0 <= index < n):  # `type() is` rejects bool indices
            continue
        if index in met_by_index:
            duplicates.add(index)
            continue
        met_by_index[index] = item.get("met") is True
    for index in duplicates:
        met_by_index[index] = False  # ambiguous double-ruling → fail closed
    return [met_by_index.get(i, False) for i in range(n)]


def _error_result(tc, message: str) -> ToolResultMessage:
    """A tool_result carrying an error string back to the model — used when a
    tool call is refused (not granted / over budget) rather than executed, so
    the model can adjust or submit instead of the turn crashing."""
    return ToolResultMessage(tool_call_id=tc.id, tool=tc.tool, content=f"error: {message}")


def _fallback_to_llm(entry: LLMFallbackEntry) -> LLMConfig:
    # LLMFallbackEntry carries no temperature/max_tokens (a fallback can't
    # itself have fallbacks either) — the resulting LLMConfig takes model
    # defaults for those, which is the intended behavior, not a silent drop.
    return LLMConfig(
        provider=entry.provider,
        model=entry.model,
        endpoint=entry.endpoint,
        api_key_ref=entry.api_key_ref,
    )
