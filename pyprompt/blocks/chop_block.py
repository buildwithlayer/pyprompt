from __future__ import annotations

from typing import Optional, Type, Tuple

from pyprompt.common.json import JSON_ARRAY, JSON_TYPE
from pyprompt.tokenizers import Tokenizer
from .block import Block

__all__ = ("ChopBlock",)


class ChopBlock(Block):
    def _build_str(self, data: str) -> str:
        return data

    def _build_array(self, data: str) -> JSON_ARRAY:
        return [data]

    def build_json(self, parent_type: Optional[Type], *args) -> JSON_TYPE:
        data = self._parse_data_from_args(*args)

        if parent_type is list:
            return self._build_array(data)
        else:
            return self._build_str(data)

    def reduce(
            self,
            tokenizer: Tokenizer,
            parent_type: Optional[Type],
            *args,
            goal: Optional[int] = None
    ) -> Tuple[str, int]:
        data = self._parse_data_from_args(*args)

        built_data = self.build_json(parent_type, data)
        built_data_size = self.size(tokenizer, built_data)

        if goal is None:
            goal = built_data_size - 1

        if built_data_size <= goal:
            return data, built_data_size

        difference = built_data_size - goal
        raw_data_size = self.size(tokenizer, data)

        if raw_data_size < difference:
            return "", 0

        raw_data_tokens = tokenizer.encode(data)
        new_data = tokenizer.decode(raw_data_tokens[:-difference])

        return new_data, len(raw_data_tokens) - difference

    @staticmethod
    def _parse_data_from_args(*args) -> str:
        if len(args) != 1:
            raise ValueError(f"Expected 1 argument, got {len(args)}")

        data = args[0]
        if not isinstance(data, str):
            raise TypeError(f"Data must be str, not: {type(data)}")

        return data
