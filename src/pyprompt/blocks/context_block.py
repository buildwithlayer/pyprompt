from __future__ import annotations

from typing import List, Optional, Type, Tuple

from pyprompt.common.json import JSON_ARRAY, JSON_TYPE
from pyprompt.tokenizers import Tokenizer
from .block import Block

__all__ = ("ContextBlock",)


class ContextBlock(Block):
    def _build_str(self, contexts: List[str]) -> str:
        return "\n".join(contexts)

    def _build_array(self, contexts: List[str]) -> JSON_ARRAY:
        return contexts

    def build_json(self, parent_type: Optional[Type], *args) -> JSON_TYPE:
        contexts = self._parse_data_from_args(*args)

        if parent_type is list:
            return self._build_array(contexts)
        else:
            return self._build_str(contexts)

    def reduce(
            self,
            tokenizer: Tokenizer,
            parent_type: Optional[Type],
            *args,
            goal: Optional[int] = None
    ) -> Tuple[List[str], int]:
        contexts = self._parse_data_from_args(*args)

        built_data = self.build_json(parent_type, contexts)
        built_data_size = self.size(tokenizer, built_data)

        if goal is None:
            goal = built_data_size - 1

        if built_data_size < goal:
            return contexts, built_data_size

        while built_data_size > goal:
            if len(contexts) == 1:
                return contexts, built_data_size

            contexts = [*contexts][:-1]
            built_data = self.build_json(parent_type, contexts)
            built_data_size = self.size(tokenizer, built_data)

        return contexts, built_data_size

    @staticmethod
    def _parse_data_from_args(*args) -> List[str]:
        if len(args) == 0:
            raise ValueError("Must provide contexts to build ContextBlock")
        elif len(args) == 1:
            if isinstance(args[0], list):
                contexts = args[0]
            else:
                contexts = [args[0]]
        else:
            contexts = list(args)

        if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
            raise TypeError(f"Contexts must be a list of str, not: {type(contexts)}")

        return contexts
