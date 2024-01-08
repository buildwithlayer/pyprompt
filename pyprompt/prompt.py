from typing import List, Optional, Dict, TypedDict
from .blocks import Block
from .types import Wrap

class Allocator:
    pass

class PromptBlock:
    block: Block
    initial_tokens: int
    wrap: Wrap

class PromptBuilder:
    def __init__(self, max_tokens: int, blocks: List[PromptBlock], allocator: Optional[Allocator] = None):
        if max_tokens is None:
            raise ValueError("max_tokens parameter is required.")
        
        self.max_tokens = max_tokens
        self.blocks = blocks
        self.allocator = allocator
        
        self._post_init()
        
    def _post_init(self):
        self._check_blocks()
        
    def _check_blocks(self):
        """
        Checks if the total number of tokens in the blocks is less than or equal to the maximum allowed tokens.
        """
            
        lst = [block["initial_tokens"] for block in self.blocks]
        total_tokens = sum(lst)
        
        if total_tokens > self.max_tokens:
            raise ValueError("The total number of tokens in the blocks is greater than the maximum allowed tokens.")
        
    def build(self) -> str:
        """
        Builds the prompt.
        """
        
        prompt = ""
        for block in self.blocks:
            
            if "block" not in block and not isinstance(block["block"], Block):
                raise ValueError("All blocks must have a 'block' key and must inherit the Block class.")
            else:
                max_tokens = block.get("initial_tokens")
                wrap = block.get("wrap", False)
                block: Block = block["block"]
                
            
            truncated_block, remaining_tokens = block.truncate(max_tokens=max_tokens, wrap=wrap)
            prompt += block.format(truncated_block, wrap=wrap)
            
        return prompt

if __name__ == "__main__":
    raise NotImplementedError(f"{__file__} This module is not meant to be executed on its own.")