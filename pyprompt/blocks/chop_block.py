
from typing import Tuple, Optional
from enum import Enum

from .block import Block
from pyprompt.tokenizers.tokenizer import Tokenizer

__all__ = ["ChopBlock"]

class ChopType(str, Enum):
    START = "start"
    END = "end"


class ChopBlock(Block[str]):
    """A block that chops data to a specified length."""

    def __init__(self, name: str, data: str, tokenizer: Optional[Tokenizer] = None, chop_type: ChopType = ChopType.END):
        """
        Initialize a ChopBlock instance.

        Args:
            data (str): The data to be chopped.
            chop_type (ChopType): The type of chopping to be performed (default: ChopType.END).
        """
        super().__init__(name, data, tokenizer)
        self.chop_type = chop_type

    def format(self, data: Optional[str] = None) -> str:
        if data is None:
            data = self.data
        
        return data
    
    def truncate(self, max_tokens: int) -> Tuple[str, int]:
        encoded = self.tokenizer.encode(self.data)
        if self.chop_type == ChopType.END:
            encoded = encoded[:max_tokens]
        else:
            encoded = encoded[-max_tokens:]

        decoded = self.tokenizer.decode(encoded)
        return decoded, len(encoded)
