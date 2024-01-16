import tiktoken
from pyprompt.blocks import ChopBlock, ChatHistoryBlock
from pyprompt.builders import Builder
import textwrap


messages = [
    {"role": "assistant", "content": "Welcome to our chat assistant! How can I assist you today?"},
    {"role": "user", "content": "I need help with this cool stuff please"},
    {"role": "assistant", "content": "Sure! What specific account settings do you need help with?"}, 
]


def test_build_prompt_chop():
    prompt_solution = [
        {"role": "system", "content": "testing the chop block"},
        {"role": "user", "content": "testing the chop block cool"},
        {"role": "system", "content": "Previous Chat Conversation:"},
        {"role": "user", "content": "I need help with this cool stuff please"},
        {"role": "assistant", "content": "Sure! What specific account settings do you need help with?"},
    ]

    # Create an instance of the PromptBuilder
    prompt = Builder().build(
        [
            {"role": "system", "content": ChopBlock("testing the chop block")},
            ChopBlock("testing the chop block " + ChopBlock("cool tho").set_tokenizer().truncate(1).build()).format(ChopBlock.Formats.MESSAGE),
            
            {"role": "system", "content": "Previous Chat Conversation:"},
            ChatHistoryBlock(messages).truncate(35),
        ]
    )
    
    # Assert that the prompt is built correctly
    assert prompt == prompt_solution

def test_build_2():
    prompt_solution = {
        "content": "testing the chop block Pretty",
        "messages": [
            {"role": "system", "content": "Pretty Weird Stuff"},
            {"role": "user", "content": "I need help with this cool stuff please"},
            {"role": "assistant", "content": "Sure! What specific account settings do you need help with?"},
        ]
    }

    # Create an instance of the PromptBuilder
    prompt = Builder().build({
        "content": "testing the chop block " + ChopBlock("Pretty Weird Stuff").set_tokenizer().truncate(1).build(),
        "messages": [
            {"role": "system", "content": "Pretty Weird Stuff"},
            ChatHistoryBlock(messages).truncate(35),
        ]
    })

    # Assert that the prompt is built correctly
    assert prompt == prompt_solution
    
def test_do_thing():
    encoder = tiktoken.get_encoding("cl100k_base")
    
    num = len(encoder.encode("user: I need help with"))
    pass
