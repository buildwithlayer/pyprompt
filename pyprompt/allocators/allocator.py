from pyprompt.tokenizers import Tokenizer

__all__ = ("Allocator",)


class Allocator:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
