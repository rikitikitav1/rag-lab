from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


# one scenario, two shapes: our client returns ChatTurn, create_agent expects a chat model.
# without this the middleware arm would go into a grid with no equivalence test behind it
class ScriptedChatModel(GenericFakeChatModel):
    turns: list = []
    seen: list = []

    def __init__(self, turns: list, **kwargs: Any):
        super().__init__(messages=iter([]), **kwargs)
        object.__setattr__(self, "turns", list(turns))
        object.__setattr__(self, "seen", [])

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        self.seen.append(messages)
        turn = self.turns.pop(0) if self.turns else {"text": "out of script"}
        calls = [
            {
                "name": call["name"],
                "args": call.get("args", {}),
                "id": call.get("id", f"call_{i}"),
            }
            for i, call in enumerate(turn.get("tool_calls") or [])
        ]
        message = AIMessage(
            content=turn.get("text") or "",
            tool_calls=calls,
            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
