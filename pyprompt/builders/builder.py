from typing import List

import tiktoken
from ..blocks import Block, Buildables, ChatHistoryBlock, ChopBlock
from ..tokenizers import Tokenizer

__all__ = ["Builder"]

class Allocator:
    pass

class Builder:
    
    def __init__(self, allocator: Allocator = None, tokenizer: Tokenizer = tiktoken.get_encoding("cl100k_base")) -> None:
        self.tokenizer = tokenizer
        self.allocator = allocator
    
    def _build_list(self, T: List[Buildables]):
        lst = []
        for _, item in enumerate(T):
            if isinstance(item, ChatHistoryBlock):
                messages = item.build()
            
                lst.extend([self.build(m) for m in messages])
            else:
                lst.append(self.build(item))
             
        return lst
    

    def build(self, T: Buildables) -> Buildables:
        if isinstance(T, list):
            return self._build_list(T)
        elif isinstance(T, dict):
            return {self.build(key): self.build(value) for key, value in T.items()}
        elif isinstance(T, Block):
            return self.build(T.build())
        elif isinstance(T, str):
            return T
        else:
            raise NotImplementedError(f"Unsupported type: {type(T)}")

