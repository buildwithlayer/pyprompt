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

    def __init__(self, data: T):
        self.data = data
        self._original_data = data
        self._has_been_formatted = False
        self._q = []
        self.tokenizer = None
        
    @staticmethod
    def _queue(func):
        def wrapper(self, *args, **kwargs):
            self._q.append((func, args, kwargs))
            return self
        return wrapper

    def format(self, to: str = None, **kwargs) -> Block[T]:
        self._has_been_formatted = True

    def set_tokenizer(self, tokenizer: Tokenizer = tiktoken.get_encoding("cl100k_base")) -> Block[T]:
        """ Usually called by the builder """
        if self.tokenizer is None:
            self.tokenizer = tokenizer
        return self
        
    def build(self) -> Buildables:
        """
        Builds the block and returns the buildables.

        Returns:
            Buildables: The buildables generated from the block.
        """
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before building")
        
        # Apply all queued methods to the data
        for func, args, kwargs in self._q:
            func(self, *args, **kwargs)
        
        return self.data
    
Buildables = Union[List, Dict, Block, str]