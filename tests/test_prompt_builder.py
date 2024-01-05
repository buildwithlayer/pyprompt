from pyprompt.prompt import PromptBuilder
from pyprompt.blocks import *

def test_build_prompt_chop():
    # Create an instance of the PromptBuilder
    prompt_builder = PromptBuilder(
        100,
        [
            (ChopBlock("Generic Content", "Some general content!"), 25),
        ],   
    )

    prompt_str = "Some general content!"

    # Assert that the prompt is built correctly
    assert prompt_builder.build() == prompt_str

prompt_str = """system: Do you think you are cool
user: like really cool?
user: no I not :("""

def test_build_prompt():
    # Create an instance of the PromptBuilder
    prompt_builder = PromptBuilder(
        100,
        [
            (ChatHistoryBlock(
                name="user_input",
                data=[
                    {"role": "system", "content": "Do you think you are cool"},
                    {"role": "user", "content": "like really cool?"},
                    {"role": "user", "content": "no I not :("},
                ],
                ), 
            50)
        ],   
    )

    # Assert that the prompt is built correctly
    assert prompt_builder.build() == prompt_str
