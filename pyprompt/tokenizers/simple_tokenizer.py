from typing import List

from .tokenizer import Tokenizer

__all__ = ["SimpleTokenizer"]


class SimpleTokenizer(Tokenizer):
    @staticmethod
    def encode(data: str) -> List[str]:
        return data.split()
    
    @staticmethod
    def decode(data: List[str]) -> str:
        return " ".join(data)
