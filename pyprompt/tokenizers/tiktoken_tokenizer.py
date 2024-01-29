from typing import List

from tiktoken import get_encoding

from .tokenizer import Tokenizer

__all__ = ("TiktokenTokenizer",)


class TiktokenTokenizer(Tokenizer):
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.enc = get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
