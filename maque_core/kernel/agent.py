"""Agent — high-level interface to the agent framework.

Wraps the ReAct loop, LLM provider, and tool registry into a single
convenient class.

Multi-turn conversation support:
- The agent maintains a persistent ``messages`` list across calls.
- Each ``ask()`` appends the user message and lets the loop append
  assistant/tool messages in-place.
- History is trimmed automatically when it exceeds ``max_history_turns``
  to prevent context overflow.
- Use ``reset()`` to start a fresh conversation.

Error propagation:
- ``ask()`` raises :class:`AgentRunError` on LLM failure by default
  (``raise_on_error=True``). This is the facade layer — callers get a
  clean exception instead of having to inspect the response.
- On failure, all messages appended during the failed call are rolled
  back so the conversation history stays clean (no orphan user message,
  no partial assistant/tool messages).
"""

from __future__ import annotations

from typing import List, Optional

from ..contracts.llm import LLMProvider
from ..contracts.messages import Message, Role
from ..contracts.tool import ToolRegistry
from .budget import RunBudget
from .errors import AgentRunError
from .loop import react_loop


class Agent:
    """High-level agent interface with multi-turn conversation memory.

    Usage::

        agent = Agent(llm=my_llm, registry=my_registry)
        answer = agent.ask("What is 2 + 2?")
        print(agent.latest_reasoning)  # the most recent reasoning

        # Multi-turn — history is preserved automatically
        answer2 = agent.ask("And what about 3 + 3?")

        # Start fresh
        agent.reset()

    Attributes:
        latest_reasoning: The reasoning content from the most recent
            LLM call. None if the last call produced no reasoning.
            Only the latest reasoning is kept — it is overwritten on
            each LLM call, not accumulated.
        messages: The full conversation history (list of Message).
            Includes system prompt, user messages, assistant responses,
            and tool results. Modified in-place by ``ask()``.
    """

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        system_prompt: str = "You are a helpful assistant.",
        max_iterations: int = 10,
        max_tool_calls: Optional[int] = None,
        verbose: bool = False,
        max_history_turns: int = 20,
        include_reasoning_in_history: bool = True,
    ):
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.verbose = verbose
        self.max_history_turns = max_history_turns
        self.include_reasoning_in_history = include_reasoning_in_history
        self.latest_reasoning: Optional[str] = None

        # Conversation history — persists across ask() calls for multi-turn.
        # Starts with just the system prompt.
        self.messages: List[Message] = [Message.system(system_prompt)]

    def ask(
        self,
        question: str,
        verbose: Optional[bool] = None,
        raise_on_error: bool = True,
    ) -> str:
        """Ask the agent a question and get an answer.

        The question is appended to the conversation history, and the
        agent's response (including any tool calls and results) is
        also appended. This means subsequent ``ask()`` calls carry
        the full conversation context.

        History is trimmed at the start of each call if it exceeds
        ``max_history_turns``. The system prompt is always preserved.

        After this call, ``self.latest_reasoning`` contains the
        reasoning from the most recent LLM call (overwritten each time,
        not accumulated across calls).

        Error handling:
            By default (``raise_on_error=True``), LLM failures raise
            :class:`AgentRunError`. When an error occurs, **all messages
            added during this call are rolled back** so the conversation
            history stays clean — no orphan user message, no partial
            assistant or tool messages.

            Pass ``raise_on_error=False`` to get the old behaviour where
            the error text is returned as the answer string.

        Args:
            question: The user's question.
            verbose: Override the agent's verbose setting for this call.
            raise_on_error: When ``True`` (default), raise
                :class:`AgentRunError` on LLM failure and roll back
                the history. When ``False``, return the error text as
                the answer (legacy behaviour).

        Returns:
            The agent's answer string.

        Raises:
            AgentRunError: When the LLM call fails and ``raise_on_error``
                is True. The conversation history is rolled back to
                its pre-call state before the exception is raised.
        """
        v = verbose if verbose is not None else self.verbose

        # Trim history before adding the new message
        self._trim_history()

        # Snapshot the current length so we can roll back on failure.
        # This fixes the P2 "history pollution" bug where a failed call
        # would leave an orphan user message in the history.
        snapshot = len(self.messages)

        # Append user message to the persistent history
        self.messages.append(Message.user(question))

        budget = RunBudget(
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
        )

        try:
            answer, reasoning = react_loop(
                llm=self.llm,
                registry=self.registry,
                messages=self.messages,
                budget=budget,
                verbose=v,
                include_reasoning_in_history=self.include_reasoning_in_history,
                raise_on_error=raise_on_error,
            )
        except Exception:
            # Roll back all messages added during this call
            # (user message + any assistant/tool messages from the loop).
            del self.messages[snapshot:]
            raise

        self.latest_reasoning = reasoning
        return answer

    def reset(self) -> None:
        """Reset the conversation history.

        Clears all messages and re-initializes with just the system
        prompt. Use this to start a fresh conversation without
        creating a new Agent instance.
        """
        self.messages = [Message.system(self.system_prompt)]
        self.latest_reasoning = None

    def _trim_history(self) -> None:
        """Trim conversation history to stay within max_history_turns.

        Strategy:
        - Always keep the system prompt (index 0).
        - Count complete "turns" in the remaining messages. A turn is
          a user message plus everything until the next user message
          (assistant response + any tool calls + tool results).
        - We reserve room for the upcoming new turn, so we trim to
          ``max_history_turns - 1`` completed turns. After the new
          user message is added and the loop runs, the total will be
          at most ``max_history_turns``.
        - If the number of completed turns exceeds the target, remove
          the oldest complete turns from the middle (between system
          prompt and the most recent turns).

        This is called automatically at the start of each ``ask()``.
        """
        if self.max_history_turns <= 0:
            return

        # Target: keep at most (max - 1) completed turns before adding the new one
        target_turns = self.max_history_turns - 1
        if target_turns <= 0:
            target_turns = 1  # always keep at least 1 turn of context if possible

        # Find all user message indices (these mark turn boundaries)
        # Skip index 0 which is the system prompt
        user_indices = [
            i for i, msg in enumerate(self.messages)
            if i > 0 and msg.role == Role.USER
        ]

        current_turns = len(user_indices)
        if current_turns <= target_turns:
            return

        # How many turns to remove from the oldest end
        turns_to_remove = current_turns - target_turns

        # The cut point is the start of the (turns_to_remove)-th user
        # message from the beginning. We keep everything from that
        # user message onward (plus the system prompt at index 0).
        if turns_to_remove < len(user_indices):
            cut_index = user_indices[turns_to_remove]
            # Keep system prompt + messages from cut_index onward
            self.messages = [self.messages[0]] + self.messages[cut_index:]

    def list_tools(self) -> List[str]:
        """Return the names of all available tools."""
        return self.registry.tool_names
