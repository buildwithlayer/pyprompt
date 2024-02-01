from typing import Optional, Type, Union

import pytest

from pyprompt.blocks import ChopBlock
from pyprompt.common.json import JSON_ARRAY
from pyprompt.tokenizers import TiktokenTokenizer


@pytest.mark.parametrize("parent_type, data, expected", [
    (
        list,
        "Hello, world!",
        ["Hello, world!"],
    ),
    (
        None,
        "Hello, world!",
        "Hello, world!",
    ),
])
def test_build_json(parent_type: Optional[Type], data: str, expected: Union[str, JSON_ARRAY]):
    chop_block = ChopBlock("test_chop_block")
    actual = chop_block.build_json(parent_type, data)
    assert actual == expected


@pytest.mark.parametrize("parent_type, data, expected", [
    (
        list,
        "Hello, world!",
        4,
    ),
    (
        None,
        "Hello, world!",
        4,
    ),
])
def test_size(parent_type: Optional[Type], data: str, expected: int):
    chop_block = ChopBlock("test_chop_block")
    tokenizer = TiktokenTokenizer()
    built_data = chop_block.build_json(parent_type, data)
    actual = chop_block.size(tokenizer, built_data)
    assert actual == expected


@pytest.mark.parametrize("parent_type, data, goal, expected_data, expected_size", [
    (
        list,
        "Hello, world!",
        4,
        "Hello, world!",
        4,
    ),
    (
        None,
        "Hello, world!",
        4,
        "Hello, world!",
        4,
    ),
    (
        list,
        "Hello, world!",
        3,
        "Hello, world",
        3,
    ),
    (
        None,
        "Hello, world!",
        3,
        "Hello, world",
        3,
    ),
    (
        list,
        "Hello, world!",
        None,
        "Hello, world",
        3,
    ),
    (
        None,
        "Hello, world!",
        None,
        "Hello, world",
        3,
    ),
])
def test_reduce(
        parent_type: Optional[Type],
        data: str,
        goal: Optional[int],
        expected_data: str,
        expected_size: int,
):
    chop_block = ChopBlock("test_chop_block")
    tokenizer = TiktokenTokenizer()
    actual_data, actual_size = chop_block.reduce(tokenizer, parent_type, data, goal=goal)
    assert actual_data == expected_data
    assert actual_size == expected_size
