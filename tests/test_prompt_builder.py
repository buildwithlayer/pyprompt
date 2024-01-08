from pyprompt.prompt import PromptBuilder
from pyprompt.blocks import *
import pytest

def test_build_prompt_chop():
    # Create an instance of the PromptBuilder
    prompt_builder = PromptBuilder(
        max_tokens=100,
        blocks=[
            {"block": ChopBlock("Generic Content", "Some general content!"),
             "initial_tokens": 25},
        ],   
    )

    prompt_str = "Some general content!"

    # Assert that the prompt is built correctly
    assert prompt_builder.build() == prompt_str

prompt_str = """Chat History:
system: Do you think you are cool
user: like really cool?
user: no I not :("""

def test_build_prompt():
    # Create an instance of the PromptBuilder
    
    chat_history_block = ChatHistoryBlock(name="user_input", data=[
            {"role": "system", "content": "Do you think you are cool"},
            {"role": "user", "content": "like really cool?"},
            {"role": "user", "content": "no I not :("},
        ]
    )
    
    prompt_builder = PromptBuilder(
        max_tokens=100,
        blocks=[
            {"block": chat_history_block, "initial_tokens": 50}
        ],   
    )

    # Assert that the prompt is built correctly
    assert prompt_builder.build() == prompt_str


def test_build_prompt_exceed_context():
    # Create an instance of the PromptBuilder
    with pytest.raises(ValueError):
        PromptBuilder(
            max_tokens=25,
            blocks=[
                {"block": ChopBlock("Generic Content", "Some general content!"), "initial_tokens": 30},
            ],   
        )