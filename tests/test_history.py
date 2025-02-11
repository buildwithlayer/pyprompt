from sys import maxsize

from pyprompt import *


def test_get_child_priority():
    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage(
            "A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"
        ),
        AssistantMessage("{answer}"),
    )
    assert prompt.get_child_priority(0) == 0
    assert prompt.get_child_priority(3) == maxsize

    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage(
            "A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"
        ),
        AssistantMessage("{answer}"),
        newest_priority=100,
    )
    assert prompt.get_child_priority(0) == 0
    assert prompt.get_child_priority(1) == 33
    assert prompt.get_child_priority(2) == 67
    assert prompt.get_child_priority(3) == 100

    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage(
            "A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"
        ),
        AssistantMessage("{answer}"),
        newest_priority=200,
        oldest_priority=100,
    )
    assert prompt.get_child_priority(0) == 100
    assert prompt.get_child_priority(1) == 133
    assert prompt.get_child_priority(2) == 167
    assert prompt.get_child_priority(3) == 200

    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage(
            "A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"
        ),
        AssistantMessage("{answer}"),
        newest_priority=200,
        step_priority=50,
    )
    assert prompt.get_child_priority(0) == 50
    assert prompt.get_child_priority(1) == 100
    assert prompt.get_child_priority(2) == 150
    assert prompt.get_child_priority(3) == 200

    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage(
            "A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"
        ),
        AssistantMessage("{answer}"),
        newest_priority=200,
        step_priority=50,
    )
    assert prompt.children[0].priority == 50
    assert prompt.children[1].priority == 100
    assert prompt.children[2].priority == 150
    assert prompt.children[3].priority == 200


def test_history_with_tools():
    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage("What's the weather today?"),
        AssistantMessage(ToolCall("some-id", "function", "check_weather", "{}")),
        ToolMessage("27 degrees Celsius", tool_call_id="some-id"),
    )
    assert len(prompt.children) == 4
    assert isinstance(prompt.children[0], UserMessage)
    assert isinstance(prompt.children[1], AssistantMessage)
    assert isinstance(prompt.children[2], UserMessage)
    assert isinstance(prompt.children[3], Chunk)
    assert isinstance(prompt.children[3].children[0], AssistantMessage)
    assert isinstance(prompt.children[3].children[1], ToolMessage)

    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage("What's the weather today?"),
        AssistantMessage(
            ToolCall("some-id", "function", "check_temperature", "{}"),
            ToolCall("some-other-id", "function", "check_rain", "{}")
        ),
        ToolMessage("27 degrees Celsius", tool_call_id="some-id"),
        ToolMessage("10% chance of rain", tool_call_id="some-other-id"),
    )
    assert len(prompt.children) == 4
    assert isinstance(prompt.children[0], UserMessage)
    assert isinstance(prompt.children[1], AssistantMessage)
    assert isinstance(prompt.children[2], UserMessage)
    assert isinstance(prompt.children[3], Chunk)
    assert isinstance(prompt.children[3].children[0], AssistantMessage)
    assert isinstance(prompt.children[3].children[1], ToolMessage)
    assert isinstance(prompt.children[3].children[2], ToolMessage)

    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage("What's the weather today?"),
        AssistantMessage(
            ToolCall("some-id", "function", "check_temperature", "{}"),
            ToolCall("some-other-id", "function", "check_rain", "{}")
        ),
        ToolMessage("27 degrees Celsius", tool_call_id="some-id"),
        ToolMessage("10% chance of rain", tool_call_id="some-other-id"),
        AssistantMessage("The temperature today is 27 degrees Celsius with a 10% chance of rain!"),
    )
    assert len(prompt.children) == 5
    assert isinstance(prompt.children[0], UserMessage)
    assert isinstance(prompt.children[1], AssistantMessage)
    assert isinstance(prompt.children[2], UserMessage)
    assert isinstance(prompt.children[3], Chunk)
    assert isinstance(prompt.children[3].children[0], AssistantMessage)
    assert isinstance(prompt.children[3].children[1], ToolMessage)
    assert isinstance(prompt.children[3].children[2], ToolMessage)
    assert isinstance(prompt.children[4], AssistantMessage)

    prompt = History(
        AssistantMessage(
            ToolCall("some-id", "function", "check_temperature", "{}"),
            ToolCall("some-other-id", "function", "check_rain", "{}")
        ),
        ToolMessage("27 degrees Celsius", tool_call_id="some-id"),
        ToolMessage("10% chance of rain", tool_call_id="some-other-id"),
        AssistantMessage("The temperature today is 27 degrees Celsius with a 10% chance of rain!"),
    )
    assert len(prompt.children) == 2
    assert isinstance(prompt.children[0], Chunk)
    assert isinstance(prompt.children[0].children[0], AssistantMessage)
    assert isinstance(prompt.children[0].children[1], ToolMessage)
    assert isinstance(prompt.children[0].children[2], ToolMessage)
    assert isinstance(prompt.children[1], AssistantMessage)
