import pytest
import timeit
from token_distribution import *
from blocks import Block

def time():
    """
    An example demonstrating how to use this module.
    """
    encoding_name = "cl100k_base"
    tokenize = tiktoken.get_encoding(encoding_name).encode
    blocks = [
        Block(name='system_prompt', data='a '*1000, to_text=str),
        Block(name='user_input', data='b ' * 1000, to_text=str),
        Block(name='chat_history', data='c '* 1000, to_text=str),
        Block(name='actions', data='d '* 1000, to_text=str),
    ]
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

    print(f'There are {tokens_remaining} tokens remaining.')

    for block in result_blocks:
        print(str(block))

    time = timeit.timeit(
    lambda: arbitrator.distribute_tokens(
        tokenize=tokenize,
        blocks=blocks,
        max_tokens=10,
    ), number=100)

    print(f'Time: {time}')

def make_blocks_helper(*args):
    return [Block(name=i, data=d, to_text=f) for (i, (d, f)) in enumerate(args)]

@pytest.fixture
def my_blocks():
    return make_blocks_helper(('a'*1000, str), ('b'*1000, str), ([1, 2, 3, 4, 5] * 200, lambda lst: ''.join(str(x) for x in lst)), ('d'*1000, str))

@pytest.fixture
def my_arbitrator():
  return Arbitrator(
    arbitration_functions=[
        ArbitrationFunction({'system_prompt'}, reduce_blocks_simple),
        ArbitrationFunction({'user_input'}, reduce_blocks_simple),
        ArbitrationFunction({'chat_history'}, reduce_blocks_simple),
        ArbitrationFunction({'actions'}, reduce_blocks_simple),
    ],
    default_function=reduce_blocks_simple
)

def are_blocks_equal(block1, block2):
    return block1.name == block2.name and block1.data == block2.data

def assert_block_lists_equal(blocks1, blocks2):
    for b1, b2 in zip(blocks1, blocks2):
        result = are_blocks_equal(b1, b2)
        if not result:
            print(f'Assertion failed: {b1.name}, length {len(b1.data)}: {b1.data} != {b2.name}, length {len(b2.data)}: {b2.data}')
        assert result

def my_tokenizer(s):
    return list(zip(s[::2], s[1::2]))