"""ReAct loop — the core Reasoning + Acting cycle.

The loop:
1. Send messages + tool schemas to LLM
2. If LLM returns text → that's the answer (done)
3. If LLM returns tool_calls → execute them, add results to messages, go to 1
4. If budget exhausted → force final answer from the LLM

Multi-turn support:
- The loop can accept an existing messages list and append to it in-place,
  so conversation history persists across multiple ``ask()`` calls.
- Reasoning from each LLM call can be stored on the assistant message
  (controlled by ``include_reasoning_in_history``), giving the model
  visibility into its own prior reasoning in subsequent turns.
- finish_reason='length' triggers one retry with a "please continue"
  system message, so truncated outputs get completed.

Error propagation:
- By default (``raise_on_error=False``), LLM errors are returned inside
  ``LLMResponse`` with ``error`` set — the legacy behaviour.
- When ``raise_on_error=True``, any LLM failure raises :class:`AgentRunError`
  instead, so the error travels through the exception channel and the
  caller doesn't have to inspect every response.
"""

from __future__ import annotations

from typing import List, Optional

from ..contracts.llm import LLMProvider
from ..contracts.messages import LLMResponse, Message, Observation, Role
from ..contracts.tool import ToolRegistry
from .budget import RunBudget
from .errors import AgentRunError


def react_loop(
    llm: LLMProvider,
    registry: ToolRegistry,
    messages: List[Message],
    budget: Optional[RunBudget] = None,
    verbose: bool = False,
    include_reasoning_in_history: bool = True,
    raise_on_error: bool = False,
) -> tuple:
    """Run the ReAct (Reasoning + Acting) loop.

    Args:
        llm: The LLM provider to use.
        registry: The tool registry to use for tool calls.
        messages: Conversation history list. The loop appends new
            messages to this list **in-place**, so callers can keep
            using the same list across multiple turns for multi-turn
            conversations. Must already contain system prompt and
            user message(s).
        budget: Optional run budget (iterations + tool calls).
        verbose: If True, print each step to stdout.
        include_reasoning_in_history: If True, store reasoning content
            on each assistant message so it's visible in future turns.
            If False, reasoning is only kept as ``latest_reasoning``
            and not persisted in history (saves tokens).
        raise_on_error: When ``False`` (default, backward-compatible),
            LLM failures are returned as an :class:`LLMResponse` with
            ``error`` set. When ``True``, they raise :class:`AgentRunError`
            instead, propagating through the exception channel.

    Returns:
        A tuple of (final_answer: str, latest_reasoning: str | None).
        latest_reasoning contains only the most recent LLM call's
        reasoning content (overwritten each iteration, not accumulated).
    """
    if budget is None:
        budget = RunBudget(max_iterations=10)

    tool_budget_exhausted = False
    latest_reasoning: Optional[str] = None

    while True:
        # Check iteration budget
        if not budget.consume_iteration():
            if verbose:
                print(f"[budget] max iterations reached ({budget.max_iterations}), forcing final answer")
            answer, reasoning = _force_final_answer(
                llm, messages, reason="max_iterations_reached",
                raise_on_error=raise_on_error,
            )
            if reasoning is not None:
                latest_reasoning = reasoning
                if include_reasoning_in_history and messages:
                    # Attach reasoning to the last assistant message
                    for msg in reversed(messages):
                        if msg.role == Role.ASSISTANT:
                            msg.reasoning = reasoning
                            break
            return answer, latest_reasoning

        # Get tool schemas
        tool_schemas = None
        if len(registry) > 0 and not tool_budget_exhausted:
            from ..contracts.tool import ToolSpec
            # Build ToolSpec list from registry
            tool_specs = []
            for name in registry.tool_names:
                spec = registry.get_spec(name)
                if spec:
                    tool_specs.append(spec)
            tool_schemas = tool_specs

        # Call LLM (with finish_reason length retry)
        response = _chat_with_length_retry(
            llm, messages, tool_schemas, budget, verbose,
            raise_on_error=raise_on_error,
        )

        # Update latest reasoning (overwrite — only keep the most recent)
        if response.reasoning is not None:
            latest_reasoning = response.reasoning

        if verbose:
            print(f"[llm] content={response.content!r} tool_calls={len(response.tool_calls)}")
            if response.reasoning:
                print(f"[llm] reasoning={response.reasoning[:80]}...")

        # Add assistant message to history
        # When include_reasoning_in_history is True, reasoning is stored
        # on the message so future turns can see the model's prior thinking.
        assistant_msg = Message.assistant(
            content=response.content,
            tool_calls=response.tool_calls,
        )
        if include_reasoning_in_history and response.reasoning is not None:
            assistant_msg.reasoning = response.reasoning
        messages.append(assistant_msg)

        # If no tool calls, we're done — return the content
        if not response.tool_calls:
            return response.content, latest_reasoning

        # If tool budget is already exhausted but model still called tools → force final answer
        if tool_budget_exhausted:
            if verbose:
                print("[budget] tool budget exhausted but model called tools, forcing final answer")
            answer, reasoning = _force_final_answer(
                llm, messages, reason="tool_budget_exhausted",
                raise_on_error=raise_on_error,
            )
            if reasoning is not None:
                latest_reasoning = reasoning
                if include_reasoning_in_history and messages:
                    for msg in reversed(messages):
                        if msg.role == Role.ASSISTANT:
                            msg.reasoning = reasoning
                            break
            return answer, latest_reasoning

        # Execute all tool calls
        for tool_call in response.tool_calls:
            # Check tool budget before executing
            if budget.max_tool_calls is not None and budget.tool_calls_used >= budget.max_tool_calls:
                tool_budget_exhausted = True
                # Add a tool message for the truncated call (Kernel layer fix:
                # budget truncation补tool消息 — ensures the model sees a
                # consistent tool_call → tool_result pair, otherwise the
                # message history has unmatched tool_calls which breaks
                # some model APIs).
                messages.append(
                    Message.tool_result(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=(
                            f"Tool call budget exhausted (max={budget.max_tool_calls}). "
                            "This call was not executed. Please provide your final answer "
                            "based on the information you already have."
                        ),
                    )
                )
                # Also add a system message explaining the budget situation
                messages.append(
                    Message.system(
                        f"Tool call budget exhausted (max={budget.max_tool_calls}). "
                        "Please provide your final answer based on the information you already have."
                    )
                )
                if verbose:
                    print(f"[budget] tool call budget exhausted at {budget.max_tool_calls}")
                break  # stop executing more tool calls this round

            observation = registry.execute(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                tool_call_id=tool_call.id,
            )
            budget.consume_tool_call()

            if verbose:
                status = "ERROR" if observation.is_error else "OK"
                print(f"[tool] {tool_call.name}({tool_call.arguments}) → {status}: {observation.content[:80]}")

            messages.append(observation.to_message())

        # Continue the loop — the next iteration will give the model
        # the tool results and let it decide what to do next


def _chat_with_length_retry(
    llm: LLMProvider,
    messages: List[Message],
    tool_schemas: Optional[list],
    budget: RunBudget,
    verbose: bool,
    raise_on_error: bool = False,
) -> LLMResponse:
    """Call the LLM once, with one retry if finish_reason is 'length'.

    When the model hits the token limit (finish_reason='length') and
    we still have iteration budget, append a system message asking the
    model to continue, then call again. This gives truncated outputs
    a chance to complete.

    Max 1 retry to prevent infinite loops.

    Error handling:
    - When ``raise_on_error`` is True and the LLM returns an error
      response (or the retry produces one), raise :class:`AgentRunError`
      instead of returning the response.
    """
    response = llm.chat(messages, tools=tool_schemas)

    # Error check — raise if enabled
    if response.error is not None and raise_on_error:
        raise AgentRunError(response.error, response=response)

    # Only retry on length truncation, and only if we have budget left
    if (
        response.finish_reason == "length"
        and budget.iterations_used < budget.max_iterations - 1
    ):
        if verbose:
            print("[llm] output truncated (finish_reason=length), retrying with continue prompt")

        # Append the truncated assistant message with a continue system prompt
        # We need to temporarily add the truncated response + continue hint
        truncated_msg = Message.assistant(
            content=response.content,
            tool_calls=response.tool_calls,
        )
        messages.append(truncated_msg)
        messages.append(
            Message.system("输出被截断，请继续生成你的回答。")
        )

        # Consume an iteration for the retry
        budget.consume_iteration()

        # Retry — this time without tools (we just want the continuation text)
        # try/finally: the temporary messages must be popped even when
        # llm.chat raises, otherwise the truncated message + continue
        # prompt leak into the caller's history.
        try:
            retry_response = llm.chat(messages, tools=None)
        finally:
            # Remove the temporary messages we added — the caller will add
            # the final assistant message itself. We combine the truncated
            # content with the continuation.
            messages.pop()  # remove system continue message
            messages.pop()  # remove truncated assistant message

        # Combine content from both calls
        combined_content = response.content + retry_response.content

        # Build combined response
        combined = LLMResponse(
            content=combined_content,
            reasoning=retry_response.reasoning or response.reasoning,
            tool_calls=retry_response.tool_calls or response.tool_calls,
            model=retry_response.model or response.model,
            usage=retry_response.usage,
            finish_reason=retry_response.finish_reason,
            error=retry_response.error,
        )

        # Error check on combined result — raise if enabled
        if combined.error is not None and raise_on_error:
            raise AgentRunError(combined.error, response=combined)

        return combined

    return response


def _force_final_answer(
    llm: LLMProvider,
    messages: List[Message],
    reason: str,
    raise_on_error: bool = False,
) -> tuple:
    """Force the model to produce a final text answer without tools.

    Called when budget is exhausted or we need to wrap up.

    Error handling:
    - When ``raise_on_error`` is True and the LLM returns an error
      response, raise :class:`AgentRunError` instead of returning.

    Returns:
        A tuple of (answer: str, reasoning: str | None).
    """
    messages.append(
        Message.system(
            f"Reached limit ({reason}). Please provide a concise final answer "
            "based on the conversation so far, without calling any more tools."
        )
    )
    response = llm.chat(messages, tools=None)

    # Error check — raise if enabled
    if response.error is not None and raise_on_error:
        raise AgentRunError(response.error, response=response)

    return response.content or "(no final answer generated)", response.reasoning
