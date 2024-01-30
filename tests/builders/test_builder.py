import pytest

from allocators import Allocator
from pyprompt.blocks import *
from pyprompt.builders import Builder
from pyprompt.builders.trees import TemplateTreeNode
from pyprompt.tokenizers import TiktokenTokenizer


@pytest.mark.parametrize("template, expected", [
    (1, None),
    (2.5, None),
    ("Hello", None),
    (["hello", "hello", "hello"], None),
    ({"hello": "world"}, None),
    (Block(name="test_block"), {"test_block": TemplateTreeNode(block=Block(name="test_block"), path=[])}),
    (
            [Block(name="test_block_0"), Block(name="test_block_1")],
            {
                "test_block_0": TemplateTreeNode(block=Block(name="test_block_0"), path=[0]),
                "test_block_1": TemplateTreeNode(block=Block(name="test_block_1"), path=[1]),
            },
    ),
    (
            {0: Block(name="test_block_0"), 1: Block(name="test_block_1")},
            {
                "test_block_0": TemplateTreeNode(block=Block(name="test_block_0"), path=[0]),
                "test_block_1": TemplateTreeNode(block=Block(name="test_block_1"), path=[1]),
            },
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
                "test_block_0": TemplateTreeNode(block=Block(name="test_block_0"), path=["another", 1]),
                "test_block_1": TemplateTreeNode(block=Block(name="test_block_1"), path=["final", "hello"]),
            },
    ),
    (f"Hello {Block(name='name')}", {"name": TemplateTreeNode(block=Block(name="name"), path=[0])})
])
def test_parse(template, expected):
    actual = Builder._parse(template)
    assert actual == expected


def test_build():
    builder = Builder(
        context_limit=60,
        allocator=Allocator(TiktokenTokenizer()),
        template={
            "messages": [
                {"role": "system", "content": ChopBlock(name="system_prompt")},
                MessagesBlock(name="messages"),
                {
                    "role": "assistant",
                    "content": f"Here is some additional context provided by the user: {ContextBlock(name='context')}",
                },
            ],
            "tools": [
                ToolsBlock(name="tools", minsize=100),
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
