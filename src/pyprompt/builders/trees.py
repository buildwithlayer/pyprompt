from __future__ import annotations

from typing import Dict, List, Union

from pyprompt.blocks import Block

__all__ = ("TemplateTreeNode", "TemplateTree")


class TemplateTreeNode:
    def __init__(self, block: Block, path: List[Union[str, int]]):
        self.block = block
        self.path = path

    def __eq__(self, other: TemplateTreeNode) -> bool:
        return self.block == other.block and self.path == other.path


TemplateTree = Dict[str, TemplateTreeNode]
