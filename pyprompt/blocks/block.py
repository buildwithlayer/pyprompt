from typing import Generic, List, Optional, Tuple, TypeVar
import tiktoken

from pyprompt.tokenizers import Tokenizer

__all__ = ["Block"]

T = TypeVar("T")


class Block(Generic[T]):
    def __init__(self, name: str, data: T, tokenizer: Optional[Tokenizer] = None):
        self.name = name
        self.data = data
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer
    
    def format(self, data: Optional[T] = None) -> str:
        raise NotImplementedError

    def truncate(self, max_tokens: int) -> Tuple[T, int]:
        raise NotImplementedError
    

