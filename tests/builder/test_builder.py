from pyprompt.blocks import ChopBlock, ChatHistoryBlock
from pyprompt.builders import Builder
import textwrap


messages = [
    {"role": "assistant", "content": "Welcome to our chat assistant! How can I assist you today?"},
    {"role": "user", "content": ChopBlock("I need help with my account settings.").truncate(4)},
    {"role": "assistant", "content": "Sure! What specific account settings do you need help with?"}, 
]


def test_build_prompt_chop():
    prompt_solution = [
        {"role": "system", "content": "testing the chop block"},
        {"role": "user", "content": "testing the chop block"},
        {"role": "system", "content": "Previous Chat Conversation:"},
        {"role": "user", "content": "I need help with"},
        {"role": "assistant", "content": "Sure! What specific account settings do you need help with?"},
    ]

    # Create an instance of the PromptBuilder
    prompt = Builder().build(
        [
            {"role": "system", "content": ChopBlock("testing the chop block")},
            ChopBlock("testing the chop block").format(ChopBlock.Formats.MESSAGE),
            
            {"role": "system", "content": "Previous Chat Conversation:"},
            ChatHistoryBlock(messages).truncate(38),
        ]
    )

    # Assert that the prompt is built correctly
    assert prompt == prompt_solution
