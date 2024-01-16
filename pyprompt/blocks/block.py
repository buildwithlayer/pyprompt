from __future__ import annotations
import stat
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
        self._q = []

        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer
        
    def format(self, to: str = None, **kwargs) -> Block[T]:
        self._has_been_formatted = True
    
    @staticmethod
    def _queue(func):
        def wrapper(self, *args, **kwargs):
            self._q.append((func, args, kwargs))
        return wrapper

    def build(self) -> Buildables:
        """
        Builds the block and returns the buildables.

        Returns:
            Buildables: The buildables generated from the block.
        """
        
        # Apply all queued methods to the data
        for func, args, kwargs in self._q:
            func(self, *args, **kwargs)
        
        return self.data
    
Buildables = Union[List, Dict, Block, str]