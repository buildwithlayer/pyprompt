from __future__ import annotations
from typing import Dict, Generic, List, Optional, TypeVar, Union
import tiktoken

from ..tokenizers import Tokenizer

__all__ = ["Block", "Buildables"]

T = TypeVar('T')

class Block(Generic[T]):
    """
    Represents a block of data with a name and associated tokenizer.

    Args:
        data (T): The data associated with the block.
        tokenizer (Optional[Tokenizer]): The tokenizer to use for the block. Defaults to None.

    Methods:
        build() -> Buildables: Builds the block and returns the buildables.
    """

    def __init__(self, data: T, tokenizer: Optional[Tokenizer] = None):
        self.data = data
        self._original_data = data
        self._has_been_formatted = False

        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer
        
    def format(self, to: str = None, **kwargs) -> Block[T]:
        self._has_been_formatted = True
        
    # @staticmethod
    # def _check_formatted(func):
    #     def wrapper(self, *args, **kwargs):
    #         if not self._has_been_formatted:
    #             raise RuntimeError("Attempted to run a method after formatting the block. This is not allowed. Please format the block last.")
    #         return func(self, *args, **kwargs)
    #     return wrapper

    def build(self) -> Buildables:
        """
        Builds the block and returns the buildables.

        Returns:
            Buildables: The buildables generated from the block.
        """
            
        return self.data
    
Buildables = Union[List, Dict, Block, str]