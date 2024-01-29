from __future__ import annotations

from enum import Enum
from typing import List, Optional, Type, Tuple

from pyprompt.common.json import JSON_TYPE
from pyprompt.common.messages import Message, message_from_dict
from pyprompt.tokenizers import Tokenizer
from .block import Block

__all__ = ("MessagesBlock", "MessagesReductionType")


class MessagesReductionType(str, Enum):
    FORGET = "forget"
    SUMMARY = "summary"


class MessagesBlock(Block):
    def __init__(self, name: str, reduce_type: MessagesReductionType = MessagesReductionType.FORGET, **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduce_type = reduce_type

    def _build_str(self, messages: List[Message]):
        return "\n".join([message.to_string() for message in messages])

    def _build_array(self, messages: List[Message]):
        return [message.to_dict() for message in messages]

    def build_json(self, parent_type: Optional[Type], *args) -> JSON_TYPE:
        messages = self._parse_data_from_args(*args)

        if parent_type is list:
            return self._build_array(messages)
        else:
            return self._build_str(messages)

    def _reduce_forget(
            self,
            tokenizer: Tokenizer,
            parent_type: Optional[Type],
            messages: List[Message],
            goal: int
    ) -> Tuple[List[Message], int]:
        built_data = self.build_json(parent_type, messages)
        built_data_size = self._size(tokenizer, built_data)

        while built_data_size > goal:
            if len(messages) == 1:
                raise ValueError(f"Cannot reduce messages [{messages}] enough to reach goal [{goal}]")

            messages = [*messages][1:]
            built_data = self.build_json(parent_type, messages)
            built_data_size = self._size(tokenizer, built_data)

        return messages, built_data_size

    def reduce(
            self,
            tokenizer: Tokenizer,
            parent_type: Optional[Type],
            *args,
            goal: Optional[int] = None
    ) -> Tuple[List[Message], int]:
        messages = self._parse_data_from_args(*args)

        built_data = self.build_json(parent_type, messages)
        built_data_size = self._size(tokenizer, built_data)

        if goal is None:
            goal = built_data_size - 1

        if built_data_size < goal:
            return messages, built_data_size

        if self.reduce_type == MessagesReductionType.FORGET:
            return self._reduce_forget(tokenizer, parent_type, messages, goal)
        else:
            # TODO: Support summarization of message history
            raise ValueError(f"Reduction type [{self.reduce_type}] is not supported")

    @staticmethod
    def _parse_data_from_args(*args) -> List[Message]:
        if len(args) == 0:
            raise ValueError("Must provide messages to build MessagesBlock")
        elif len(args) == 1:
            if isinstance(args[0], list):
                messages = args[0]
            else:
                messages = [args[0]]
        else:
            messages = list(args)

        if not isinstance(messages, list):
            raise TypeError(f"Messages must be a list of dicts or Messages, not: {type(messages)}")
        for idx, message in enumerate(messages):
            if isinstance(message, Message):
                continue
            elif isinstance(message, dict):
                messages[idx] = message_from_dict(message)
            else:
                raise TypeError(f"Messages must be a list of dicts or Messages, not: {type(message)}")

        return messages
