from inspect import isclass

import pytest

from allocators import Allocator
from pyprompt.blocks import *
from pyprompt.builders import Builder


def assert_parsed(actual, expected):
    if actual is None or expected is None:
        assert actual == expected
    else:
        if isinstance(actual, dict):
            assert type(actual) is type(expected)

            actual_keys = actual.keys()
            expected_keys = expected.keys()
            assert set(actual_keys) == set(expected_keys)

            for key, actual_value in actual.items():
                expected_value = expected[key]
                assert_parsed(actual_value, expected_value)
        elif isclass(actual) and issubclass(actual, Block):
            assert issubclass(actual, expected)
        else:
            assert actual == expected


@pytest.mark.parametrize("template, expected", [
    (1, None),
    (2.5, None),
    ("Hello", None),
    (["hello", "hello", "hello"], None),
    ({"hello": "world"}, None),
    (Block(name="test_block"), {"test_block": {"block_type": Block, "path": []}}),
    (
            [Block(name="test_block_0"), Block(name="test_block_1")],
            {"test_block_0": {"block_type": Block, "path": [0]}, "test_block_1": {"block_type": Block, "path": [1]}},
    ),
    (
            {0: Block(name="test_block_0"), 1: Block(name="test_block_1")},
            {"test_block_0": {"block_type": Block, "path": [0]}, "test_block_1": {"block_type": Block, "path": [1]}},
    ),
    (
            {
                "hello": "world",
                "another": [
                    "hello",
                    Block(name="test_block_0"),
                ],
                "final": {
                    "hello": Block(name="test_block_1"),
                },
            }, {
                "test_block_0": {"block_type": Block, "path": ["another", 1]},
                "test_block_1": {"block_type": Block, "path": ["final", "hello"]},
            },
    ),
    (f"Hello {Block(name='name')}", {"name": {"block_type": Block, "path": [0]}})
])
def test_parse(template, expected):
    actual = Builder._parse(template)
    assert_parsed(actual, expected)


def test_build():
    builder = Builder(
        context_limit=0,
        allocator=Allocator(),
        template={
            "messages": [
                {"role": "system", "content": ChopBlock(name="system_prompt")},
                MessagesBlock(name="messages", minsize=300),
                {
                    "role": "assistant",
                    "content": f"Here is some additional context provided by the user: {ContextBlock(name='context', minsize=200)}",
                },
            ],
            "tools": [
                ToolsBlock(name="tools", minsize=300),
            ],
        }
    )

    prompt = builder.build(
        system_prompt="Keep your responses brief.",
        messages=[
            {
                "role": "assistant",
                "content": "How can I help you today?",
            },
            {
                "role": "user",
                "content": "What is the weather today?",
            },
        ],
        context=[
            "The temperature outside is 60 degrees Fahrenheit."
        ],
        tools=[],
    )

    assert prompt == {
        "messages": [
            {
                "role": "system",
                "content": "Keep your responses brief.",
            },
            {
                "role": "assistant",
                "content": "How can I help you today?",
            },
            {
                "role": "user",
                "content": "What is the weather today?",
            },
            {
                "role": "assistant",
                "content": "Here is some additional context provided by the user: The temperature outside is 60 degrees Fahrenheit.",
            },
        ],
        "tools": [],
    }
