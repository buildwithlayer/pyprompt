import tiktoken
import logging
import os

from pyprompt.arbitrator import Arbitrator, ArbitrationFunction, Block, reduce_blocks_simple

logger = logging.getLogger(os.path.basename(__file__))

# For demonstration purposes
example_blocks = [
    Block(name="system_prompt", data="aa aaa", to_text=str),
    Block(name="user_input", data="bb bbb", to_text=str),
    Block(name="chat_history", data="cc ccc", to_text=str),
    Block(name="actions", data="dd ddd", to_text=str),
]

# For demonstration purposes
example_arbitrator = Arbitrator(
    arbitration_functions=[
        ArbitrationFunction({"system_prompt", "user_input"}, reduce_blocks_simple),
        ArbitrationFunction({"chat_history"}, reduce_blocks_simple),
        ArbitrationFunction({"actions"}, reduce_blocks_simple),
    ],
    default_function=reduce_blocks_simple,
)

def example():
    """
    An example demonstrating how to use this module.
    """
    encoding_name = "cl100k_base"

    tokenize = tiktoken.get_encoding(encoding_name).encode

    tokens_remaining, result_blocks = example_arbitrator.distribute_tokens(
        tokenize=tokenize,
        blocks=example_blocks,
        max_tokens=10,
    )

    logger.info(f' There are {tokens_remaining} tokens remaining.')

    for block in result_blocks:
        logger.info(f' Block: {block}')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example()
