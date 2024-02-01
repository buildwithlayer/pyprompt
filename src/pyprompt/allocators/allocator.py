from pyprompt.blocks import Block
from pyprompt.common.json import JSON_TYPE
from pyprompt.tokenizers import Tokenizer
from .trees import *

__all__ = ("Allocator",)


class Allocator:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def get_template_size(self, template: JSON_TYPE) -> int:
        if isinstance(template, str):
            # TODO: don't include the fstring braces
            return len(self.tokenizer.encode(template))
        elif isinstance(template, list):
            return sum(self.get_template_size(part) for part in template)
        elif isinstance(template, dict):
            return sum(self.get_template_size(part) for part in template.values())
        elif isinstance(template, Block):
            return 0
        else:
            return self.get_template_size(str(template))

    def reduce(self, allocator_tree: AllocatorTree, desired_change: int):
        changed = 0

        while True:
            nodes_that_can_shrink = len([node for node in allocator_tree.values() if node.can_shrink])
            if nodes_that_can_shrink == 0:
                raise ValueError("Cannot shrink nodes")

            shrink_amount_per_node = max(1, round((desired_change - changed) / nodes_that_can_shrink))

            for node in allocator_tree.values():
                if not node.can_shrink:
                    continue

                goal = node.size - shrink_amount_per_node
                if goal < 0:
                    goal = 0

                args, size = node.block.reduce(
                    self.tokenizer,
                    node.parent_type,
                    node.args,
                    goal=goal,
                )
                if node.minsize is None or size >= node.minsize:
                    changed += (node.size - size)
                    if size > goal or goal == 0:
                        node.can_shrink = False
                    node.args = args
                    node.size = size

                    if changed >= desired_change:
                        return

    @staticmethod
    def satisfied(allocator_tree: AllocatorTree) -> bool:
        for node in allocator_tree.values():
            if node.maxsize is not None and node.maxsize < node.size:
                return False
        return True
