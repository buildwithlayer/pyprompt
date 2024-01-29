from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal, List, Union

from .tools import *

__all__ = ("MessageRole", "SystemMessage", "UserMessage", "AssistantMessage", "ToolMessage", "Message", "message_from_dict")


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class BaseMessage:
    def to_dict(self) -> dict:
        return {
            "role": self.role.value
        }

    def to_string(self) -> str:
        raise NotImplementedError


@dataclass
class SystemMessage(BaseMessage):
    content: str
    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM
    name: Optional[str] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        return d

    def to_string(self) -> str:
        if self.name is not None:
            return f"{self.name} ({self.role.value.upper()}): {self.content}"
        else:
            return f"{self.role.value.upper()}: {self.content}"


@dataclass
class UserMessage(BaseMessage):
    content: str
    role: Literal[MessageRole.USER] = MessageRole.USER
    name: Optional[str] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        return d

    def to_string(self) -> str:
        if self.name is not None:
            return f"{self.name} ({self.role.value.upper()}): {self.content}"
        else:
            return f"{self.role.value.upper()}: {self.content}"


@dataclass
class AssistantMessage(BaseMessage):
    content: Optional[str] = None
    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.content:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls is not None:
            d["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return d

    def to_string(self) -> str:
        lines = []
        if self.content is not None:
            if self.name is not None:
                lines.append(f"{self.name} ({self.role.value.upper()}): {self.content}")
            else:
                lines.append(f"{self.role.value.upper()}: {self.content}")
        if self.tool_calls is not None:
            for tool_call in self.tool_calls:
                if self.name is not None:
                    lines.append(f"{self.name} ({self.role.value.upper()}): {tool_call.to_string()}")
                else:
                    lines.append(f"{self.role.value.upper()}: {tool_call.to_string()}")
        return "\n".join(lines)


@dataclass
class ToolMessage(BaseMessage):
    content: str
    tool_call_id: str
    role: Literal[MessageRole.TOOL] = MessageRole.TOOL

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["content"] = self.content
        d["tool_call_id"] = self.tool_call_id
        return d

    def to_string(self) -> str:
        return f"{self.role.value.upper()} ({self.tool_call_id}): {self.content}"


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


def message_from_dict(message_dict: dict) -> Message:
    role = message_dict.pop("role", None)
    if role is None:
        raise ValueError("Message dict must have a role")

    if role == MessageRole.SYSTEM:
        return SystemMessage(**message_dict)
    elif role == MessageRole.USER:
        return UserMessage(**message_dict)
    elif role == MessageRole.ASSISTANT:
        return AssistantMessage(**message_dict)
    elif role == MessageRole.TOOL:
        return ToolMessage(**message_dict)
    else:
        raise ValueError(f"Message role must be a valid MessageRole, not: {role}")
