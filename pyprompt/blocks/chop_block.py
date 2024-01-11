from __future__ import annotations
from typing import Tuple, Optional
from enum import Enum

from .block import Block, Buildables
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

    def __init__(self, data: str, tokenizer: Optional[Tokenizer] = None):
        super().__init__(data, tokenizer)

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

    # Block._check_formatted
    def truncate(self, max_tokens: int) -> ChopBlock:
        """
        Truncates the data in the block to a maximum number of tokens.
        """
        encoded = self.tokenizer.encode(self.data)

        if self.chop_type == ChopType.END:
            encoded = encoded[:max_tokens]
        else:
            encoded = encoded[-max_tokens:]

        self.data = self.tokenizer.decode(encoded)
        
        return self
