from __future__ import annotations
from pyexpat.errors import messages
from typing import Tuple, Optional
from enum import Enum

from tiktoken.registry import get_encoding as get_encoding

from pyprompt.tokenizers import Tokenizer

from .block import Block
from ..tokenizers import Tokenizer

__all__ = ["ChopBlock"]


class ChopType(str, Enum):
    START = "start"
    END = "end"


class ChopBlock(Block[str]):
    """
    A block that represents a piece of text that can be chopped or formatted.

    Args:
        data (str): The initial data for the block.
        tokenizer (Optional[Tokenizer]): The tokenizer to use for encoding and decoding the data.

    Attributes:
        Formats (Enum): An enumeration of the available formats for the block.
    """

    class Formats(Enum):
        STRING = "string"
        MESSAGE = "message"

    def __init__(self, data: str):
        super().__init__(data)

    def format(self, to: ChopBlock.Formats = None, **kwargs) -> ChopBlock:
        """
        Formats the data in the block.

        Args:
            to (ChopBlock.Formats): The format to convert the data to. Default is ChopBlock.Formats.STRING.
            **kwargs: Additional keyword arguments for formatting.

        Note:
            This method should always be called last, after any other modifications to the data.

        Example:
            >>> block = ChopBlock("Hello, world!")
            >>> block.truncate(50).format(ChopBlock.Formats.MESSAGE, role="user")
        """

        if to == None:
            to = ChopBlock.Formats.STRING

        if to == ChopBlock.Formats.STRING:
            self.data = str(self.data)
        elif to == ChopBlock.Formats.MESSAGE:
            self.data = {"role": kwargs.get("role", "user"), "content": self.data}

        return self

    @Block._queue
    def truncate(self, max_tokens: int, **kwargs) -> ChopBlock:
        """
        Truncates the data in the block to a maximum number of tokens.
        """
        encoded = self.tokenizer.encode(self.data)

        if kwargs.get("chop_type", ChopType.END) == ChopType.END:
            encoded = encoded[:max_tokens]
        else:
            encoded = encoded[-max_tokens:]

        self.data = self.tokenizer.decode(encoded)

        return self
    
    def set_tokenizer(self, *args) -> ChopBlock:
        return super().set_tokenizer(*args)