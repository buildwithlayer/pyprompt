import math
from importlib import import_module
from copy import deepcopy
from json import loads, JSONDecodeError
from string import Formatter
from typing import List, Optional, Union, Any, Tuple, Type

from pyprompt.allocators import Allocator
from pyprompt.allocators.trees import AllocatorTreeNode, AllocatorTree
from pyprompt.blocks import Block
from pyprompt.common.json import *
from .trees import *

__all__ = ("Builder",)


class Builder:
    def __init__(self, context_limit: int, allocator: Allocator, template: JSON_TYPE):
        self.context_limit = context_limit
        self.allocator = allocator
        self.template = template

        self.template_tree = self._parse_template_tree(template)

    def build(self, **kwargs) -> JSON_TYPE:
        template = deepcopy(self.template)

        expected_block_names = self.template_tree.keys()
        actual_block_names = kwargs.keys()

        if set(expected_block_names) != set(actual_block_names):
            raise KeyError(f"Expected block names {expected_block_names}, got {actual_block_names}")

        actual_context_limit = self.context_limit - self.allocator.get_template_size(self.template)
        if actual_context_limit <= 0:
            raise ValueError(f"Template size too large for context limit")

        size_tree = self._parse_size_tree(actual_context_limit)

        start_size = 0
        for block_name, block_map_value in self.template_tree.items():
            template, template_type, size = self._fill_block(
                template,
                block_name,
                block_map_value.block,
                block_map_value.path,
                kwargs[block_name],
            )

            node = size_tree[block_name]
            node.parent_type = template_type
            node.size = size
            node.args = kwargs[block_name]
            if node.maxsize is not None and node.maxsize < node.size:
                node.args, node.size = node.block.reduce(
                    self.allocator.tokenizer,
                    template_type,
                    node.args,
                    goal=node.maxsize,
                )
            node.can_shrink = node.minsize is None or node.size > node.minsize

            start_size += node.size

        self.allocator.reduce(size_tree, start_size - actual_context_limit)

        template = deepcopy(self.template)
        for block_name, block_map_value in self.template_tree.items():
            template, template_type, size = self._fill_block(
                template,
                block_name,
                block_map_value.block,
                block_map_value.path,
                size_tree[block_name].args,
            )

        return template

    def _parse_size_tree(self, actual_context_limit: int) -> AllocatorTree:
        size_tree = dict()
        for block_name, block_map_value in self.template_tree.items():
            minsize = block_map_value.block.minsize
            if isinstance(minsize, float):
                minsize = int(math.floor(actual_context_limit * minsize))

            maxsize = block_map_value.block.maxsize
            if isinstance(minsize, float):
                maxsize = int(math.floor(actual_context_limit * maxsize))

            size_tree[block_name] = AllocatorTreeNode(block_map_value.block, minsize, maxsize)
        return size_tree

    def _fill_block(
            self,
            template: JSON_TYPE,
            block_name: str,
            block: Block,
            block_path: List[Union[str, int]],
            args: Any,
    ) -> Tuple[JSON_TYPE, Optional[Type], int]:
        if len(block_path) == 0:
            block_data = block.build_json(None, args)
            return block_data, None, block.size(self.allocator.tokenizer, block_data)
        elif len(block_path) == 1:
            template_type = type(template)
            block_data = block.build_json(template_type, args)
            block_size = block.size(self.allocator.tokenizer, block_data)

            if template_type == str:
                formatter = Formatter()
                parsed_args = list(formatter.parse(template))
                arg_name = parsed_args[block_path[0]][1]
                return (
                    template.replace("{" + arg_name + "}", f"{block_data}"),
                    template_type,
                    block_size,
                )
            if template_type == list and isinstance(block_data, list):
                template.pop(block_path[0])
                for _block_item in reversed(block_data):
                    template.insert(block_path[0], _block_item)
                return template, template_type, block_size
            else:
                template[block_path[0]] = block_data
                return template, template_type, block_size
        else:
            sub_template = template[block_path[0]]
            sub_block_path = block_path[1:]
            template[block_path[0]], template_type, size = self._fill_block(
                sub_template, block_name, block, sub_block_path, args,
            )
            return template, template_type, size

    @staticmethod
    def _parse_template_tree(template: JSON_TYPE) -> TemplateTree:
        parsed = Builder._parse(template)
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid tempalte:\n{template}")
        return parsed

    @staticmethod
    def _parse(value: Union[Block, JSON_TYPE]) -> Optional[TemplateTree]:
        if value is None or isinstance(value, JSON_NUMBER.__args__) or isinstance(value, bool):
            return None
        elif isinstance(value, str):
            formatter = Formatter()
            parsed_args = list(formatter.parse(value))

            if len(parsed_args) == 0:
                return None

            if all(_arg[1] is None for _arg in parsed_args):
                return None

            d = dict()
            for idx, parsed_value in enumerate(parsed_args):
                try:
                    block_data = loads(parsed_value[1])
                except JSONDecodeError:
                    continue

                if not isinstance(block_data, list):
                    continue

                block_dict = block_data[0]
                if not isinstance(block_dict, dict):
                    continue

                block_module_name = block_dict.pop("module", None)
                if not isinstance(block_module_name, str):
                    continue

                block_class_name = block_dict.pop("class", None)
                if not isinstance(block_class_name, str):
                    continue

                block_module = import_module(block_module_name)
                block_class = getattr(block_module, block_class_name)
                # TODO: verify block_class is subclass of block

                block = block_class(**block_dict)

                parsed = Builder._parse(block)
                if not parsed:
                    continue

                d.update(Builder._add_value(idx, parsed))

            return d if len(d) > 0 else None
        elif isinstance(value, Block):
            return {value.name: TemplateTreeNode(block=value, path=[])}
        elif isinstance(value, list):
            d = dict()
            for idx, v in reversed(list(enumerate(value))):
                parsed = Builder._parse(v)
                if parsed:
                    for _k, _v in parsed.items():
                        if _k in d:
                            raise ValueError(f"Duplicate block name: {_k}")
                    d.update(Builder._add_value(idx, parsed))
            return d if len(d) > 0 else None
        elif isinstance(value, dict):
            d = dict()
            for key, v in value.items():
                parsed = Builder._parse(v)
                if parsed:
                    for _k, _v in parsed.items():
                        if _k in d:
                            raise ValueError(f"Duplicate block name: {_k}")
                    d.update(Builder._add_value(key, parsed))
            return d if len(d) > 0 else None
        else:
            raise TypeError(f"Unexpected type: {type(value)}")

    @staticmethod
    def _add_value(current_identifier: Union[str, int], current_parsed: TemplateTree) -> TemplateTree:
        return {
            key: TemplateTreeNode(
                block=value.block,
                path=[current_identifier] + value.path,
            ) for key, value in current_parsed.items()
        }
