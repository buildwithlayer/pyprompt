import tiktoken
import logging
from .blocks import Block
from typing import List, Tuple, Optional

class Allocator:
    pass

class PromptBuilder:
    def __init__(self, max_tokens, blocks: List[Tuple[Block, int]], allocator: Optional[Allocator] = None):
        self.max_tokens = max_tokens
        self.blocks = blocks
        self.allocator = allocator
        
    def _post_init(self):
        if not self._check_blocks():
            raise ValueError("The total number of tokens in the blocks is greater than the maximum allowed tokens.")
        
    def _check_blocks(self):
        """
        Checks if the total number of tokens in the blocks is less than or equal to the maximum allowed tokens.
        """
        total_tokens = sum([block[1] for block in self.blocks])
        return total_tokens <= self.max_tokens
        
    def build(self) -> str:
        """
        Builds the prompt.
        """
        
        prompt = ""
        for (block, i) in self.blocks:
            truncted_block = block.truncate(max_tokens=i)
            prompt += block.format(truncted_block)
            
        return prompt

if __name__ == "__main__":
    raise NotImplementedError(f"{__file__} This module is not meant to be executed on its own.")