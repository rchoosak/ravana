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

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ravana.compiler.graph import CompiledGraph
from ravana.runtime.base import AgentTurnResult, TransientAgentError
from ravana.runtime.idempotency import compute_idempotency_key
from ravana.runtime.prompt import assemble_system_prompt
from ravana.runtime.providers.base import (
    AssistantMessage,
    Capability,
    Message,
    ProviderAdapter,
    ProviderError,
    ProviderRequest,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from ravana.schema.models import AgentConfig, LLMConfig, LLMFallbackEntry

SUBMIT_RESULT = "submit_result"

# §3.6: per-fallback-entry retry budget — deliberately small (1) and separate
# from the engine-level guards.max_retries_per_node, so a long fallback chain
# can't multiply total attempts.
_PER_ENTRY_RETRIES = 1

# A schema-less fallback when a node declares no output_schema: submit_result
# then accepts any object, and the whole returned object becomes the
# state_delta. (A stricter default could require declared state keys; kept
# permissive here so agents without output_schema still work.)
_ANY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}


class ToolExecutor(Protocol):
    """Executes a real (non-submit_result) tool call and returns a result
    string to feed back into the turn. Toolkits (§1.7) implement this in the
    next slice; Phase 0b's gateway slice injects a no-op/fake.

    `idempotency_key` is the content-addressed key from §3.6, computed by the
    gateway and passed in **before** execution — a side-effecting connector
    (git push, ticket create) MUST dedupe on it, which is only possible if it
    has the key at execution time, not after. The engine's later persistence
    of the same key into message.tool_calls (loop.py) uses this same value."""

    async def execute(
        self, *, run_id: str, node_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str
    ) -> str: ...


class _NoToolExecutor:
    async def execute(
        self, *, run_id: str, node_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str
    ) -> str:
        raise ProviderError(
            f"agent tried to call tool '{tool}' but no ToolExecutor is wired "
            "(toolkits are the next Phase 0b slice)"
        )


@dataclass
class _Strategy:
    """The resolved structured-output approach for one agent (§3.4), decided
    once from the provider's capabilities rather than per call."""

    use_guided: bool
    use_native_forced_tool: bool
    # If neither of the above, the only lever left is the repair loop.


def _select_strategy(adapter: ProviderAdapter, model: str) -> _Strategy:
    caps = adapter.capabilities(model)
    if Capability.GUIDED_DECODING in caps:
        return _Strategy(use_guided=True, use_native_forced_tool=False)
    if Capability.NATIVE_STRUCTURED_OUTPUT in caps:
        return _Strategy(use_guided=False, use_native_forced_tool=True)
    return _Strategy(use_guided=False, use_native_forced_tool=False)


def _validate(payload: Any, output_schema: dict[str, Any] | None) -> str | None:
    """Returns an error string if payload violates the (shallow) schema, else
    None. Deliberately minimal for Phase 0a/0b — checks type, required keys,
    and top-level enums, which is what the §4 example's schemas use. A full
    JSON Schema validator (jsonschema) is a drop-in upgrade later."""
    if output_schema is None:
        return None if isinstance(payload, dict) else "expected a JSON object"
    if output_schema.get("type") == "object" and not isinstance(payload, dict):
        return "expected a JSON object"
    for key in output_schema.get("required", []):
        if key not in payload:
            return f"missing required field '{key}'"
    for key, spec in output_schema.get("properties", {}).items():
        if key in payload and "enum" in spec and payload[key] not in spec["enum"]:
            return f"field '{key}' must be one of {spec['enum']}, got {payload[key]!r}"
    return None


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


class LLMGateway:
    def __init__(
        self,
        graph: CompiledGraph,
        adapters: dict[str, ProviderAdapter],
        tool_executor: ToolExecutor | None = None,
    ):
        self._graph = graph
        self._adapters = adapters
        self._tools = tool_executor or _NoToolExecutor()
        # §3.4: strategy is decided once per (provider, model), not re-derived
        # every call. Capabilities are static, so this memo is the "decided at
        # registration" contract without a separate registration step.
        self._strategy_cache: dict[tuple[str, str], _Strategy] = {}

    def _adapter_for(self, provider: str) -> ProviderAdapter:
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise ProviderError(f"no adapter registered for provider '{provider}'")
        return adapter

    def _strategy_for(self, adapter: ProviderAdapter, model: str) -> _Strategy:
        key = (adapter.name, model)
        if key not in self._strategy_cache:
            self._strategy_cache[key] = _select_strategy(adapter, model)
        return self._strategy_cache[key]

    async def run_turn(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        agent_id: str,
        shared_state: dict[str, Any],
    ) -> AgentTurnResult:
        agent = self._graph.agents_by_id[agent_id]
        system = assemble_system_prompt(agent, self._graph.skills_by_id, shared_state)
        output_schema = agent.output_schema or _ANY_OBJECT_SCHEMA

        # §3.6: try the primary llm, then each fallback entry, each with its
        # own small retry budget (default 1 retry per entry — distinct from,
        # and smaller than, the engine-level max_retries_per_node, so a chain
        # of N fallbacks can't multiply total attempts by N). A provider-level
        # failure (ProviderError) is what a retry/fallback responds to; a
        # TransientAgentError from repair-budget exhaustion is the model
        # producing bad output, not a provider fault, so it propagates
        # immediately without burning the fallback chain. Only when every
        # entry's budget is spent does the node_execution fail.
        chain: list[LLMConfig] = [agent.llm, *(_fallback_to_llm(f) for f in agent.llm.fallback)]
        last_error: Exception | None = None
        for llm in chain:
            for _ in range(_PER_ENTRY_RETRIES + 1):
                try:
                    return await self._run_one_llm(
                        run_id=run_id, node_id=node_id, agent=agent, llm=llm,
                        system=system, output_schema=output_schema,
                    )
                except ProviderError as exc:
                    last_error = exc
                    continue
        raise TransientAgentError(f"all LLM entries exhausted for agent '{agent_id}': {last_error}")

    async def _run_one_llm(
        self,
        *,
        run_id: str,
        node_id: str,
        agent: AgentConfig,
        llm: LLMConfig,
        system: str,
        output_schema: dict[str, Any],
    ) -> AgentTurnResult:
        adapter = self._adapter_for(llm.provider)
        strategy = self._strategy_for(adapter, llm.model)
        if strategy.use_guided:
            # Guided decoding constrains the *entire* response to the schema,
            # so the payload arrives as JSON in the message text (not a
            # submit_result tool call) and there's no intermediate tool loop —
            # a guided completion is one-shot by construction. This is the read
            # path the earlier code lacked: it waited for a submit_result tool
            # even under guided decoding, so a real guided runtime's schema
            # JSON would have looked like "no tool call".
            return await self._run_guided(node_id=node_id, llm=llm, adapter=adapter, system=system, output_schema=output_schema)
        return await self._run_tool_loop(
            run_id=run_id, node_id=node_id, llm=llm, adapter=adapter, system=system, output_schema=output_schema
        )

    async def _run_guided(self, *, node_id, llm, adapter, system, output_schema):
        guards = self._graph.doc.spec.graph.guards
        messages: list[Message] = [UserMessage(text="Respond with your final structured output as JSON.")]
        input_tokens = output_tokens = repair_count = 0
        while True:
            request = ProviderRequest(
                model=llm.model, system=system, messages=messages, output_schema=output_schema,
                temperature=llm.temperature, max_tokens=llm.max_tokens, endpoint=llm.endpoint, api_key_ref=llm.api_key_ref,
            )
            response = await adapter.complete(request)
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
                raise TransientAgentError(f"node '{node_id}' guided output invalid after {repair_count} repairs: {error}")
            repair_count += 1
            messages.append(UserMessage(text=f"That output was invalid: {error}. Return valid JSON matching the schema."))

    async def _run_tool_loop(self, *, run_id, node_id, llm, adapter, system, output_schema):
        guards = self._graph.doc.spec.graph.guards
        submit_tool = Tool(
            name=SUBMIT_RESULT,
            description="Call this exactly once, when done, to submit your final structured result.",
            input_schema=output_schema,
        )
        messages: list[Message] = [
            UserMessage(text="Complete your task, then call submit_result with your structured output.")
        ]
        tool_call_count = repair_count = input_tokens = output_tokens = 0
        recorded_tool_calls: list[dict[str, Any]] = []

        while True:
            # §3.4.4: submit_result is offered as a tool and the prompt asks the
            # model to call it, but tool_choice is force-set to it ONLY once the
            # tool budget is spent — forcing every turn would drop the model's
            # free-text reasoning (§3.4.2). Forcing at exhaustion still
            # guarantees the turn terminates rather than looping or going quiet.
            force_submit = tool_call_count >= guards.max_tool_calls_per_turn
            request = ProviderRequest(
                model=llm.model,
                system=system,
                messages=messages,
                tools=[submit_tool],
                force_tool=SUBMIT_RESULT if force_submit else None,
                temperature=llm.temperature,
                max_tokens=llm.max_tokens,
                endpoint=llm.endpoint,
                api_key_ref=llm.api_key_ref,
            )
            response = await adapter.complete(request)
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens

            submit = next((tc for tc in response.tool_calls if tc.tool == SUBMIT_RESULT), None)
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
                        tool_call_count=tool_call_count,
                    )
                if repair_count >= guards.max_output_repairs:
                    raise TransientAgentError(
                        f"node '{node_id}' output failed validation after {repair_count} repairs: {error}"
                    )
                repair_count += 1
                messages.append(UserMessage(text=f"Your submit_result was invalid: {error}. Try again."))
                continue

            if not response.tool_calls:
                messages.append(UserMessage(text="Call submit_result now with your final output."))
                tool_call_count += 1
                continue

            # Preserve the assistant's tool_calls turn in the transcript before
            # the results — real providers reject a tool result with no
            # preceding assistant tool_calls message. Then execute each tool,
            # computing its content-addressed idempotency key (§3.6) BEFORE the
            # call so a side-effecting connector can dedupe a retry.
            messages.append(AssistantMessage(tool_calls=response.tool_calls, text=response.text))
            for tc in response.tool_calls:
                tool_call_count += 1
                key = compute_idempotency_key(run_id, node_id, tc.tool, tc.arguments)
                result = await self._tools.execute(
                    run_id=run_id, node_id=node_id, tool=tc.tool, arguments=tc.arguments, idempotency_key=key
                )
                recorded_tool_calls.append({"tool": tc.tool, "arguments": tc.arguments, "idempotency_key": key})
                messages.append(ToolResultMessage(tool_call_id=tc.id, tool=tc.tool, content=result))


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
