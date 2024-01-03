import tiktoken
from pyprompt.arbitrator import Arbitrator, ArbitrationFunction, Block, reduce_blocks_simple
import timeit
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))

def time_example():
    """
    An example demonstrating how to time this module.
    """
    encoding_name = "cl100k_base"
    tokenize = tiktoken.get_encoding(encoding_name).encode
    
    blocks = [
        Block(name='system_prompt', data='a '*1000, to_text=str),
        Block(name='user_input', data='b ' * 1000, to_text=str),
        Block(name='chat_history', data='c '* 1000, to_text=str),
        Block(name='actions', data='d '* 1000, to_text=str),
    ]
    
    # Names in sets correspond to block names
    arbitrator = Arbitrator(
        arbitration_functions=[
            ArbitrationFunction({'system_prompt'}, reduce_blocks_simple),
            ArbitrationFunction({'user_input'}, reduce_blocks_simple),
            ArbitrationFunction({'chat_history'}, reduce_blocks_simple),
            ArbitrationFunction({'actions'}, reduce_blocks_simple),
        ],
        default_function=reduce_blocks_simple
    )
    tokens_remaining, result_blocks = arbitrator.distribute_tokens(
        tokenize=tokenize,
        blocks=blocks,
        max_tokens=3000,
    )

    logger.info(f'There are {tokens_remaining} tokens remaining.')

    for block in result_blocks:
        logger.info((str(block)))

    time = timeit.timeit(
    lambda: arbitrator.distribute_tokens(
        tokenize=tokenize,
        blocks=blocks,
        max_tokens=10,
    ), number=100)

    logger.info(f'Time: {time}')
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    time_example()