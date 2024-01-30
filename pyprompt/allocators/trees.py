from __future__ import annotations

from typing import Optional, Type, Any, Dict

from pyprompt.blocks import Block

__all__ = ("AllocatorTreeNode", "AllocatorTree")


class AllocatorTreeNode:
    def __init__(
            self,
            block: Block,
            minsize: Optional[int],
            maxsize: Optional[int],
            parent_type: Optional[Type] = None,
            args: Any = None,
    ):
        self.block = block
        self.minsize = minsize
        self.maxsize = maxsize
        self.parent_type = parent_type
        self.args = args
        self.size = self.maxsize
        self.can_shrink = True
        self.done = False

    def __eq__(self, other: AllocatorTreeNode) -> bool:
        return all([
            self.block == other.block,
            self.minsize == other.minsize,
            self.maxsize == other.maxsize,
            self.parent_type == other.parent_type,
            self.args == other.args,
            self.size == other.size,
            self.can_shrink == other.can_shrink,
            self.done == other.done,
        ])


AllocatorTree = Dict[str, AllocatorTreeNode]
