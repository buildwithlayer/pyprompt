from typing import List

__all__ = ("Tokenizer",)


class Tokenizer:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError
