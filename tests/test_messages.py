from conftest import encode

from pyprompt import *


def test_message_get_token_count():
    message = Message("Hello, world!", role="user")
    token_map = {
        0: [9906, 11, 1917, 0],
    }
    assert message.get_token_count(token_map) == 8


def test_message_get_messages():
    message = Message("Hello, world!", role="user")
    render_map = {
        0: "Hello, world!",
    }
    assert list(message.generate_messages(render_map)) == [
        {
            "role": "user",
            "content": "Hello, world!",
        }
    ]


def test_tool_call_dict():
    tool_call = ToolCall("some-id", "function", "ReallyCoolFunction", "{}")
    assert tool_call.__dict__ == {
        "id": "some-id",
        "type": "function",
        "function": {
            "name": "ReallyCoolFunction",
            "arguments": "{}",
        },
    }


def test_tool_call_get_token_map():
    tool_call = ToolCall("some-id", "function", "ReallyCoolFunction", "{}")
    assert tool_call.get_token_map(dict(), encode) == {
        0: [
            5018,
            307,
            794,
            330,
            15031,
            13193,
            498,
            330,
            1337,
            794,
            330,
            1723,
            498,
            330,
            1723,
            794,
            5324,
            609,
            794,
            330,
            49885,
            57850,
            5263,
            498,
            330,
            16774,
            794,
            36603,
            32075,
        ],
    }


def test_tool_call_get_token_count():
    tool_call = ToolCall("some-id", "function", "ReallyCoolFunction", "{}")
    token_map = {
        0: [
            5018,
            307,
            794,
            330,
            15031,
            13193,
            498,
            330,
            1337,
            794,
            330,
            1723,
            498,
            330,
            1723,
            794,
            5324,
            609,
            794,
            330,
            49885,
            57850,
            5263,
            498,
            330,
            16774,
            794,
            36603,
            32075,
        ],
    }
    assert tool_call.get_token_count(token_map) == 29


def test_assistant_message_generate_messages():
    message = AssistantMessage("Hello, how can I help you?")
    render_map = {
        0: "Hello, how can I help you?",
    }
    assert list(message.generate_messages(render_map)) == [
        {
            "role": "assistant",
            "content": "Hello, how can I help you?",
        }
    ]

    message = AssistantMessage(
        ToolCall("some-id", "function", "ReallyCoolFunction", "{}")
    )
    render_map = {
        0: {
            0: '{"id": "some-id", "type": "function", "function": {"name": "ReallyCoolFunction", "arguments": "{}"}}',
        },
    }
    assert list(message.generate_messages(render_map)) == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "some-id",
                    "type": "function",
                    "function": {
                        "name": "ReallyCoolFunction",
                        "arguments": "{}",
                    },
                }
            ],
        }
    ]


def test_tool_message_generate_messages():
    message = ToolMessage("The tool call was successful!", tool_call_id="some-id")
    render_map = {
        0: "The tool call was successful!",
    }
    assert list(message.generate_messages(render_map)) == [
        {
            "role": "tool",
            "content": "The tool call was successful!",
            "tool_call_id": "some-id",
        }
    ]
