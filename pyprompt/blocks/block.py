from typing import Generic, List, Optional, Tuple, TypeVar
import tiktoken

from pyprompt.tokenizers import Tokenizer

__all__ = ["Block"]

T = TypeVar("T")


class Block(Generic[T]):
    """
    Represents a block of data with a name and associated tokenizer.

    Args:
        name (str): The name of the block.
        data (T): The data associated with the block.
        tokenizer (Optional[Tokenizer]): The tokenizer to use for the block. Defaults to None.

    Methods:
        format(data: Optional[T] = None) -> str: Formats the block's data into a string representation.
        truncate(max_tokens: int) -> Tuple[T, int]: Truncates the block's data to a specified maximum number of tokens.
    """
    def __init__(self, name: str, data: T, tokenizer: Optional[Tokenizer] = None):
        self.name = name
        self.data = data
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer
    
    def format(self, data: Optional[T] = None) -> str:
        """
        Formats the block's data into a string representation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement format()")

    def truncate(self, max_tokens: int) -> Tuple[T, int]:
        """
        Truncates the block's data to a specified maximum number of tokens.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement format()")
    

