from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Type, Tuple

from pyprompt.common.json import JSON_TYPE
from pyprompt.common.tools import Tool, tool_from_dict
from pyprompt.tokenizers import Tokenizer
from .block import Block

__all__ = ("ToolsBlock",)


class ToolsBlock(Block):
    def _build_str(self, tools: List[Tool]):
        return "\n".join([tool.to_string() for tool in tools])

    def _build_array(self, tools: List[Tool]):
        return [tool.to_dict() for tool in tools]

    def build_json(self, parent_type: Optional[Type], *args) -> JSON_TYPE:
        tools = self._parse_data_from_args(*args)

        if parent_type is list:
            return self._build_array(tools)
        else:
            return self._build_str(tools)

    def reduce(
            self,
            tokenizer: Tokenizer,
            parent_type: Optional[Type],
            *args,
            goal: Optional[int] = None
    ) -> Tuple[List[Tool], int]:
        # TODO: How do we want to reduce tools?
        raise NotImplementedError

    @staticmethod
    def _parse_data_from_args(*args) -> List[Tool]:
        if len(args) == 0:
            raise ValueError("Must provide tools to build ToolsBlock")
        elif len(args) == 1:
            if isinstance(args[0], list):
                tools = args[0]
            else:
                tools = [args[0]]
        else:
            tools = args

        tools = deepcopy(tools)

        if not isinstance(tools, list):
            raise TypeError(f"Tools must be a list of dicts or Tools, not: {type(tools)}")
        for idx, tool in enumerate(tools):
            if isinstance(tool, Tool):
                continue
            elif isinstance(tool, dict):
                tools[idx] = tool_from_dict(tool)
            else:
                raise TypeError(f"Tools must be a list of dicts or Tools, not: {type(tool)}")

        return tools
