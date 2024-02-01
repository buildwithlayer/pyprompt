from dataclasses import dataclass
from typing import Literal, Optional

__all__ = ("Tool", "ToolCall", "tool_from_dict")


@dataclass
class Tool:
    @dataclass
    class Function:
        name: str
        description: Optional[str] = None
        parameters: Optional[dict] = None

        def to_dict(self) -> dict:
            d = {
                "name": self.name,
            }
            if self.description is not None:
                d["description"] = self.description
            if self.parameters is not None:
                d["parameters"] = self.parameters
            return d

        def to_string(self) -> str:
            lines = [f"{self.name} Function"]
            if self.description is not None:
                lines.append(self.description)
            if self.parameters is not None:
                lines.extend(["Parameters:", self.parameters])
            return "\n".join(lines)

    function: Function
    type: Literal["function"] = "function"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "function": self.function.to_dict(),
        }

    def to_string(self) -> str:
        return self.function.to_string()


@dataclass
class ToolCall:
    @dataclass
    class Function:
        name: str
        arguments: str

        def to_dict(self) -> dict:
            return {
                "name": self.name,
                "arguments": self.arguments,
            }

        def to_string(self) -> str:
            return f"Calling {self.name} with arguments: {self.arguments}"

    id: str
    function: Function
    type: Literal["function"] = "function"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "function": self.function.to_dict(),
            "type": self.type,
        }

    def to_string(self) -> str:
        return self.function.to_string()


def tool_from_dict(tool_dict: dict) -> Tool:
    tool_function_dict: Optional[dict] = tool_dict.pop("function", None)
    if tool_function_dict is None:
        raise ValueError("Tool dict must have function")

    return Tool(function=Tool.Function(**tool_function_dict), **tool_dict)
