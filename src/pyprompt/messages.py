import json
from typing import Generator

from .prompt_element import PromptElement
from .utils import DecodingFunc, EncodingFunc, RenderMap, TokenMap

__all__ = (
    "Message",
    "AssistantMessage",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "UserMessage",
)


class Message(PromptElement):
    def __init__(
        self,
        *children: PromptElement | str,
        role: str = "",
        **kwargs,
    ):
        super().__init__(*children, **kwargs)

        if role not in ("assistant", "user", "tool", "system"):
            raise ValueError("role must be 'assistant', 'user', 'tool', or 'system'")
        self.role = role

    def get_token_count(self, token_map: TokenMap) -> int:
        return super().get_token_count(token_map) + 4

    def generate_messages(self, render_map: RenderMap) -> Generator[dict, None, None]:
        message = {"role": self.role}
        for value in self.iter_render_map(render_map):
            if isinstance(value, str):
                if "content" in message:
                    message["content"] += value
                else:
                    message["content"] = value
        yield message


class ToolCall(PromptElement):
    def __init__(
        self, id: str, type: str, function_name: str, function_arguments: str, **kwargs
    ):
        super().__init__(id, type, function_name, function_arguments, **kwargs)
        self.id = id
        self.type = type
        self.function_name = function_name
        self.function_arguments = function_arguments

    @property
    def __dict__(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function_name,
                "arguments": self.function_arguments,
            },
        }

    def get_token_map(self, props: dict, encoding_func: EncodingFunc) -> TokenMap:
        return {0: encoding_func(json.dumps(self.__dict__))}

    def get_token_count(self, token_map: TokenMap) -> int:
        return sum(len(v) for v in token_map.values())

    def prune(
        self,
        budget: int,
        token_map: TokenMap,
        encoding_func: EncodingFunc,
        decoding_func: DecodingFunc,
    ) -> TokenMap | None:
        return None

    def iter_render_map(
        self, render_map: RenderMap
    ) -> Generator[str | PromptElement, None, None]:
        yield self


class AssistantMessage(Message):
    def __init__(self, *children: PromptElement | str, **kwargs):
        super().__init__(*children, role="assistant", **kwargs)

    def get_token_count(self, token_map: TokenMap) -> int:
        token_count = super().get_token_count(token_map)
        if any(isinstance(child, str) for child in self.children) and any(
            isinstance(child, ToolCall) for child in self.children
        ):
            token_count += 2
        return token_count

    def generate_messages(self, render_map: RenderMap) -> Generator[dict, None, None]:
        message = {"role": self.role}
        for value in self.iter_render_map(render_map):
            if isinstance(value, str):
                if "content" in message:
                    message["content"] += value
                else:
                    message["content"] = value
            elif isinstance(value, ToolCall):
                if "tool_calls" not in message:
                    message["tool_calls"] = []
                message["tool_calls"].append(value.__dict__)
        yield message

    @property
    def has_tool_calls(self) -> bool:
        return any(isinstance(child, ToolCall) for child in self.children)


class SystemMessage(Message):
    def __init__(self, *children: PromptElement | str, **kwargs):
        super().__init__(*children, role="system", **kwargs)


class ToolMessage(Message):
    def __init__(self, *children: PromptElement | str, tool_call_id: str, **kwargs):
        super().__init__(*children, role="tool", **kwargs)
        self.tool_call_id = tool_call_id

    def get_token_count(self, token_map: TokenMap) -> int:
        return super().get_token_count(token_map) + 2

    def generate_messages(self, render_map: RenderMap) -> Generator[dict, None, None]:
        for message in super().generate_messages(render_map):
            message["tool_call_id"] = self.tool_call_id
            yield message


class UserMessage(Message):
    def __init__(self, *children: PromptElement | str, **kwargs):
        super().__init__(*children, role="user", **kwargs)
