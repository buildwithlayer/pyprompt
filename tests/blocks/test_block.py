import importlib
from string import Formatter

import json

from pyprompt.blocks import Block


def test_block_format():
    actual_fstring = f"Some block is here: {Block(name='Test')}"
    expected_fstring = "Some block is here: {[{\"module\": \"pyprompt.blocks.block\", \"class\": \"Block\", \"name\": \"Test\"}]}"
    assert actual_fstring == expected_fstring

    formatter = Formatter()
    actual_parse = list(formatter.parse(expected_fstring))
    expected_parse = [("Some block is here: ", "[{\"module\": \"pyprompt.blocks.block\", \"class\": \"Block\", \"name\": \"Test\"}]", '', None)]
    assert actual_parse == expected_parse

    actual_data = json.loads(expected_parse[0][1])
    expected_data = [{"module": "pyprompt.blocks.block", "class": "Block", "name": "Test"}]
    assert actual_data == expected_data

    block_module = importlib.import_module(expected_data[0]["module"])
    block_class = getattr(block_module, expected_data[0]["class"])
    actual_block = block_class(name=expected_data[0]["name"])
    expected_block = Block(name='Test')
    assert actual_block.name == expected_block.name
