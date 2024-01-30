from __future__ import annotations

import json
from typing import Type, Optional, Union, Tuple, Any

from pyprompt.common.json import JSON_ARRAY, JSON_OBJECT, JSON_TYPE
from pyprompt.tokenizers import Tokenizer

__all__ = ("Block",)


class Block:
    def __init__(
            self,
            name: str,
            minsize: Optional[Union[int, float]] = None,
            maxsize: Optional[Union[int, float]] = None,
            **kwargs,
    ):
        self.name = name
        self.minsize = minsize
        self.maxsize = maxsize

    def __format__(self, _) -> str:
        return "{[" + json.dumps(self.to_kwargs()) + "]}"

    def to_kwargs(self) -> dict:
        d = {
            "module": self.__module__,
            "class": self.__class__.__name__,
            "name": self.name,
        }
        if self.minsize is not None:
            d["minsize"] = self.minsize
        if self.maxsize is not None:
            d["maxsize"] = self.maxsize
        return d

    def _build_str(self, *args) -> str:
        raise NotImplementedError

    def _build_array(self, *args) -> JSON_ARRAY:
        raise NotImplementedError

    def _build_dict(self, *args) -> JSON_OBJECT:
        raise NotImplementedError

    def build_json(self, parent_type: Optional[Type], *args) -> JSON_TYPE:
        if parent_type == str:
            return self._build_str(*args)
        elif parent_type == list:
            return self._build_array(*args)
        else:
            return self._build_dict(*args)

    def reduce(
            self,
            tokenizer: Tokenizer,
            parent_type: Optional[Type],
            *args,
            goal: Optional[int] = None
    ) -> Tuple[Any, int]:
        raise NotImplementedError

    @staticmethod
    def size(tokenizer: Tokenizer, built_data: Union[str, JSON_ARRAY, JSON_OBJECT]) -> int:
        if isinstance(built_data, str):
            return len(tokenizer.encode(built_data))
        elif isinstance(built_data, list):
            return sum(Block.size(tokenizer, datum) for datum in built_data)
        elif isinstance(built_data, dict):
            return sum(Block.size(tokenizer, datum) for datum in built_data.values())
        else:
            raise TypeError(f"Unexpected type {type(built_data)}")

    @staticmethod
    def _parse_data_from_args(*args) -> Any:
        raise NotImplementedError

    def __eq__(self, other: Block) -> bool:
        return all([
            type(self) is type(other),
            self.name == other.name,
            self.minsize == other.minsize,
            self.maxsize == other.maxsize,
        ])
