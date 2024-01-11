from typing import List
from pyprompt.blocks.chat_history_block import ChatHistoryBlock
from ..blocks import Block, Buildables

__all__ = ["Builder"]

class Builder:
    
    
    def _build_list(self, T: List[Buildables]):
        lst = []
        for _, item in enumerate(T):
            if isinstance(item, ChatHistoryBlock):
                lst.extend(item.build())
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

