import tiktoken
import logging
from pyprompt.arbitrator import example_arbitrator
from pyprompt.prompt import Prompt
from pyprompt.blocks import Block

# For demonstration purposes
example_blocks = [
    Block(name="system_prompt", data="aaa", to_text=str),
    Block(name="user_input", data="bbb", to_text=str),
    Block(name="chat_history", data="ccc", to_text=str),
    Block(name="actions", data="ddd", to_text=str),
]

def example():
    """
    An example demonstrating how to use this module.
    """
    encoding_name = "cl100k_base"

    tokenize = tiktoken.get_encoding(encoding_name).encode

    prompt = Prompt(
        blocks=example_blocks,
        arbitrator=example_arbitrator,
        tokenize=tokenize,
        max_tokens=10,
    )

    print(prompt)

if __name__ == "__main__":
    raise NotImplementedError(f"{__file__} This module is not meant to be executed on its own.")