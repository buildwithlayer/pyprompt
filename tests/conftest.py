import pytest
import tiktoken

from pyprompt import *


def encode(txt: str) -> list[int]:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(txt)


def decode(tokens: list[int]) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)


@pytest.fixture
def generic_prompt() -> PromptElement:
    return PromptElement(
        SystemMessage(
            "You are a helpful assistant that cheers people up.", priority=1000
        ),
        SystemMessage("How are you?", priority=900),
        SystemMessage("I am fantastic. How are you?", priority=900),
        SystemMessage("What time is it?", priority=100),
        SystemMessage("It's high time to be happy!", priority=100),
        SystemMessage("What is your name?", priority=700),
        SystemMessage("My name is Happy Copilot.", priority=700),
        UserMessage("Hello, how are you?", priority=499),
        AssistantMessage("I am terrific, how are you?", priority=500),
        UserMessage("What time is it?", priority=900),
    )
