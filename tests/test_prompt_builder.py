from pyprompt.prompt import PromptBuilder
from pyprompt.blocks import *
import pytest
import textwrap

def test_build_prompt_chop():
    # Create an instance of the PromptBuilder
    prompt_builder = PromptBuilder(
        max_tokens=100,
        blocks=[
            {"block": ChopBlock("Product Description", "Introducing our new product!"),
             "initial_tokens": 25},
        ],   
    )

    prompt_str = "Introducing our new product!"

    # Assert that the prompt is built correctly
    assert prompt_builder.build() == prompt_str


messages = [
    {"role": "customer", "content": "Do you have this product in stock?"},
    {"role": "salesperson", "content": "Yes, we have it available."},
    {"role": "customer", "content": "Great! I would like to purchase one."},
]

prompt_str = """Chat History:
customer: Do you have this product in stock?
salesperson: Yes, we have it available.
customer: Great! I would like to purchase one."""

def test_build_prompt():
    # Create an instance of the PromptBuilder
    
    prompt_builder = PromptBuilder(
        max_tokens=100,
        blocks=[
            {"block": ChatHistoryBlock(name="user_input", data=messages), "initial_tokens": 50, "wrap": True}
        ],   
    )

    x = prompt_builder.build()
    
    y = textwrap.dedent(prompt_str)
    # Assert that the prompt is built correctly
    assert prompt_builder.build() == prompt_str.replace("\t", "")
    
def test_build_prompt_mixed_blocks():
    
    
    prompt_str = """Some general
Chat History:
customer: Do you have this product in stock?
salesperson: Yes, we have it available.
customer: Great! I would like to purchase one."""

    # Create an instance of the PromptBuilder
    prompt_builder = PromptBuilder(
        max_tokens=100,
        blocks=[
            {"block": ChopBlock("Generic Content", "Some general content!"), "initial_tokens": 2, "wrap": True},
            {"block": ChatHistoryBlock("Chat History", data=messages), "initial_tokens": 50, "wrap": True}
        ],   
    )

    # Assert that the prompt is built correctly
    assert prompt_builder.build() == textwrap.dedent(prompt_str)
    


def test_build_prompt_exceed_context():
    # Create an instance of the PromptBuilder
    with pytest.raises(ValueError):
        PromptBuilder(
            max_tokens=25,
            blocks=[
                {"block": ChopBlock("Generic Content", "Some general content!"), "initial_tokens": 30},
            ],   
        )