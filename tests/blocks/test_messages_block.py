from typing import Optional, Type, Union, List

import pytest

from pyprompt.blocks import MessagesBlock
from pyprompt.common.json import JSON_ARRAY
from pyprompt.common.messages import Message, AssistantMessage, UserMessage
from pyprompt.tokenizers import TiktokenTokenizer


@pytest.mark.parametrize("parent_type, messages, expected", [
    (
        list,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?"),
        ],
        [
            {'content': 'Hello! How can I help you?', 'role': 'assistant'},
            {'content': 'What time is it?', 'role': 'user'},
        ],
    ),
    (
        None,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?"),
        ],
        "ASSISTANT: Hello! How can I help you?\nUSER: What time is it?",
    ),
])
def test_build_json(
        parent_type: Optional[Type],
        messages: List[Union[Message, dict]],
        expected: Union[str, JSON_ARRAY],
):
    messages_block = MessagesBlock("test_messages_block")
    actual = messages_block.build_json(parent_type, *messages)
    assert actual == expected


@pytest.mark.parametrize("parent_type, messages, expected", [
    (
        list,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?"),
        ],
        15,
    ),
    (
        None,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?"),
        ],
        19,
    ),
])
def test_size(parent_type: Optional[Type], messages: List[Union[Message, dict]], expected: int):
    messages_block = MessagesBlock("test_messages_block")
    tokenizer = TiktokenTokenizer()
    actual = messages_block.size(tokenizer, parent_type, *messages)
    assert actual == expected


@pytest.mark.parametrize("parent_type, messages, goal, expected_data, expected_size", [
    (
        list,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?"),
        ],
        15,
        [
            AssistantMessage(content="Hello! How can I help you?"),
            UserMessage(content="What time is it?"),
        ],
        15,
    ),
    (
        None,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?")
        ],
        19,
        [
            AssistantMessage(content="Hello! How can I help you?"),
            UserMessage(content="What time is it?"),
        ],
        19,
    ),
    (
        list,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?"),
        ],
        14,
        [
            UserMessage(content="What time is it?"),
        ],
        6,
    ),
    (
        None,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?")
        ],
        18,
        [
            UserMessage(content="What time is it?"),
        ],
        7,
    ),
    (
        list,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?"),
        ],
        None,
        [
            UserMessage(content="What time is it?"),
        ],
        6,
    ),
    (
        None,
        [
            {"role": "assistant", "content": "Hello! How can I help you?"},
            UserMessage(content="What time is it?")
        ],
        None,
        [
            UserMessage(content="What time is it?"),
        ],
        7,
    ),
])
def test_reduce(
        parent_type: Optional[Type],
        messages: List[Union[Message, dict]],
        goal: Optional[int],
        expected_data: str,
        expected_size: int,
):
    messages_block = MessagesBlock("test_messages_block")
    tokenizer = TiktokenTokenizer()
    actual_data, actual_size = messages_block.reduce(tokenizer, parent_type, *messages, goal=goal)
    assert actual_data == expected_data
    assert actual_size == expected_size
