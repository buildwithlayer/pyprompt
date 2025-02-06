from pyprompt import *


def test_get_child_priority():
    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage("A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"),
        AssistantMessage("{answer}"),
    )
    assert prompt.get_child_priority(0) == 0
    assert prompt.get_child_priority(3) == maxsize

    prompt = History(
        UserMessage("Hello!"),
        AssistantMessage("Hello, how can I help you?"),
        UserMessage("A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"),
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
        UserMessage("A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"),
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
        UserMessage("A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"),
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
        UserMessage("A runaway trolley car is careening down a track. Five people stand in its path, unaware of the imminent threat. You stand at the intersection of two different tracks and could, if you chose to, divert the trolley onto another track where only one person would be killed. Do you divert the trolley, intentionally killing one to save five?"),
        AssistantMessage("{answer}"),
        newest_priority=200,
        step_priority=50,
    )
    assert prompt.children[0].priority == 50
    assert prompt.children[1].priority == 100
    assert prompt.children[2].priority == 150
    assert prompt.children[3].priority == 200
