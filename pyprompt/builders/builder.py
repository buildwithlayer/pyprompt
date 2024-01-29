from importlib import import_module
from copy import deepcopy
from json import loads, JSONDecodeError
from string import Formatter
from typing import Dict, List, Optional, Union, TypedDict, Type, Any

from pyprompt.allocators import Allocator
from pyprompt.blocks import Block
from pyprompt.common.json import *

__all__ = ("Builder",)


class _BlockMapValue(TypedDict):
    block_type: Type[Block]
    path: List[Union[str, int]]


_BlockMap = Dict[str, _BlockMapValue]


class Builder:
    def __init__(self, context_limit: int, allocator: Allocator, template: JSON_TYPE):
        self.context_limit = context_limit
        self.allocator = allocator
        self.template = template

        self.blocks = self._parse_template(template)

    def build(self, **kwargs) -> JSON_TYPE:
        template = deepcopy(self.template)

        expected_block_names = self.blocks.keys()
        actual_block_names = kwargs.keys()

        if set(expected_block_names) != set(actual_block_names):
            raise KeyError(f"Expected block names {expected_block_names}, got {actual_block_names}")

        for block_name, block_map_value in self.blocks.items():
            template = self._fill_block(
                template,
                block_name,
                block_map_value["block_type"],
                block_map_value["path"],
                kwargs[block_name],
            )

        return template

    @staticmethod
    def _fill_block(
            template: JSON_TYPE,
            block_name: str,
            block_type: Type[Block],
            block_path: List[Union[str, int]],
            args: Any,
    ) -> JSON_TYPE:
        if len(block_path) == 0:
            return block_type(name=block_name).build_json(None, args)
        elif len(block_path) == 1:
            template_type = type(template)
            block_data = block_type(name=block_name).build_json(template_type, args)

            if template_type == str:
                formatter = Formatter()
                parsed_args = list(formatter.parse(template))
                arg_name = parsed_args[block_path[0]][1]
                return template.replace("{" + arg_name + "}", f"{block_data}")
            if template_type == list and isinstance(block_data, list):
                template.pop(block_path[0])
                for _block_item in reversed(block_data):
                    template.insert(block_path[0], _block_item)
                return template
            else:
                template[block_path[0]] = block_data
                return template
        else:
            sub_template = template[block_path[0]]
            sub_block_path = block_path[1:]
            template[block_path[0]] = Builder._fill_block(sub_template, block_name, block_type, sub_block_path, args)
            return template

    @staticmethod
    def _parse_template(template: JSON_TYPE) -> _BlockMap:
        parsed = Builder._parse(template)
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid tempalte:\n{template}")
        return parsed

    @staticmethod
    def _parse(value: Union[Block, JSON_TYPE]) -> Optional[_BlockMap]:
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
            block_class = type(f"{value.name}_block", (value.__class__,), value.to_kwargs())
            if not issubclass(block_class, Block):
                raise ValueError(f"Unexpected type {type(block_class)}")
            return {value.name: _BlockMapValue(block_type=block_class, path=[])}
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
    def _add_value(current_identifier: Union[str, int], current_parsed: _BlockMap) -> _BlockMap:
        return {
            key: _BlockMapValue(
                block_type=value["block_type"],
                path=[current_identifier] + value["path"]
            ) for key, value in current_parsed.items()
        }
