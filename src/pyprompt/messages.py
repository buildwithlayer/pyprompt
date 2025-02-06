from typing import Generator

from .prompt_element import PromptElement
from .utils import RenderMap, TokenMap, iter_render_map

__all__ = ("Message", "AssistantMessage", "SystemMessage", "UserMessage")


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
        content = ""
        for value in iter_render_map(render_map):
            content += value
        yield {"role": self.role, "content": content}


class AssistantMessage(Message):
    def __init__(self, *children: PromptElement | str, **kwargs):
        super().__init__(*children, role="assistant", **kwargs)


class SystemMessage(Message):
    def __init__(self, *children: PromptElement | str, **kwargs):
        super().__init__(*children, role="system", **kwargs)


class UserMessage(Message):
    def __init__(self, *children: PromptElement | str, **kwargs):
        super().__init__(*children, role="user", **kwargs)
