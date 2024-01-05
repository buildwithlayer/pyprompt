from typing import List

from .tokenizer import Tokenizer

__all__ = ["SimpleTokenizer"]


class SimpleTokenizer(Tokenizer):
    @staticmethod
    def encode(data: str) -> List[int]:
        return [ord(ch) for ch in data]
    
    @staticmethod
    def decode(data: List[int]) -> str:
        return "".join([chr(i) for i in data])
