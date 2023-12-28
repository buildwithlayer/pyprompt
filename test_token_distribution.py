from token_distribution import *
import timeit
import pytest

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
            print(f'Assertion failed: {b1.name}: {b1.data} != {b2.name}: {b2.data}')
        assert result

# Test order blocks

def test_order_blocks_empty():
    assert_block_lists_equal([], order_blocks([], []))

def test_order_blocks_forward():
    order_by = [
        Block(name='system_prompt', data='a', to_text=str),
        Block(name='user_input', data='b', to_text=str),
        Block(name='chat_history', data='c', to_text=str),
    ]
    to_order = {
        Block(name='system_prompt', data='d', to_text=str),
        Block(name='user_input', data='e', to_text=str),
        Block(name='chat_history', data='f', to_text=str),
    }
    expected = [
        Block(name='system_prompt', data='d', to_text=str),
        Block(name='user_input', data='e', to_text=str),
        Block(name='chat_history', data='f', to_text=str),
    ]
    assert_block_lists_equal(expected, order_blocks(order_by, to_order))

def test_order_blocks_reverse():
    order_by = [
        Block(name='chat_history', data='c', to_text=str),
        Block(name='user_input', data='b', to_text=str),
        Block(name='system_prompt', data='a', to_text=str),

    ]
    to_order = {
        Block(name='system_prompt', data='d', to_text=str),
        Block(name='user_input', data='e', to_text=str),
        Block(name='chat_history', data='f', to_text=str),
    }
    expected = [
        Block(name='chat_history', data='f', to_text=str),
        Block(name='user_input', data='e', to_text=str),
        Block(name='system_prompt', data='d', to_text=str),
    ]
    assert_block_lists_equal(expected, order_blocks(order_by, to_order))

# Test reduce single block

def my_tokenizer(s):
    return list(zip(s[::2], s[1::2]))

def test_reduce_single_block_zero_tokens():
    b = Block(name='system_prompt', data='abcdef', to_text=str)
    expected = Block(name='system_prompt', data='', to_text=str)
    actual_tokens_remaining, actual = reduce_single_block(my_tokenizer, b, 0)
    assert are_blocks_equal(expected, actual)
    assert 0 == actual_tokens_remaining

def test_reduce_single_block_short_tokens():
    b = Block(name='system_prompt', data='abcdef', to_text=str)
    expected = Block(name='system_prompt', data='cdef', to_text=str)
    actual_tokens_remaining, actual = reduce_single_block(my_tokenizer, b, 2)
    assert are_blocks_equal(expected, actual)
    assert 0 == actual_tokens_remaining

def test_reduce_single_block_exact_tokens():
    b = Block(name='system_prompt', data='abcdef', to_text=str)
    actual_tokens_remaining, actual = reduce_single_block(my_tokenizer, b, 3)
    assert are_blocks_equal(b, actual)
    assert 0 == actual_tokens_remaining

def test_reduce_single_block_excess_tokens():
    b = Block(name='system_prompt', data='abcdef', to_text=str)
    actual_tokens_remaining, actual = reduce_single_block(my_tokenizer, b, 5)
    assert are_blocks_equal(b, actual)
    assert 2 == actual_tokens_remaining

def test_reduce_single_block_empty():
    b = Block(name='system_prompt', data='', to_text=str)
    actual_tokens_remaining, actual = reduce_single_block(my_tokenizer, b, 5)
    assert are_blocks_equal(b, actual)
    assert 5 == actual_tokens_remaining

# Test distribute tokens

def test_distribute_tokens_zero_tokens(my_blocks, my_arbitrator):
    expected = make_blocks_helper(('', str), ('', str), ([], lambda lst: ''.join(str(x) for x in lst)), ('', str))
    tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(my_tokenizer, my_blocks, 0)
    assert_block_lists_equal(expected, result_blocks)
    assert 0 == tokens_remaining

def test_distribute_tokens_correct_tokens(my_blocks, my_arbitrator):
    tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(my_tokenizer, my_blocks, 2000)
    assert_block_lists_equal(my_blocks, result_blocks)
    assert 0 == tokens_remaining

def test_distrubite_tokens_excess_tokens(my_blocks, my_arbitrator):
    tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(my_tokenizer, my_blocks, 2077)
    assert_block_lists_equal(my_blocks, result_blocks)
    assert 77 == tokens_remaining

def test_distribute_tokens_dearth_of_tokens(my_blocks, my_arbitrator):
    expected = make_blocks_helper(('a'*998, str), ('', str), ([], lambda lst: ''.join(str(x) for x in lst)), ('', str))
    tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(my_tokenizer, my_blocks, 499)
    assert_block_lists_equal(expected, result_blocks)
    assert 0 == tokens_remaining

def test_distribute_tokens_zero_blocks(my_arbitrator):
    tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(my_tokenizer, [], 0)
    assert_block_lists_equal([], result_blocks)
    assert 0 == tokens_remaining

def test_distribute_tokens_one_block():
    pass

def test_distribute_tokens_in_arbitrators_ooo(my_arbitrator):
    blocks = [
        Block(name='system_prompt', data='aaa', to_text=str),
        Block(name='chat_history', data='bbb', to_text=str),
        Block(name='user_input', data='ccc', to_text=str),
        Block(name='actions', data='ddd', to_text=str),
        Block(name='not_in_arbitrators', data='eee', to_text=str),
    ]
    expected = [
        Block(name='system_prompt', data='aaa', to_text=str),
        Block(name='chat_history', data='', to_text=str),
        Block(name='user_input', data='cc', to_text=str),
        Block(name='actions', data='', to_text=str),
        Block(name='not_in_arbitrators', data='', to_text=str),
    ]
    tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(list, blocks, 5)
    assert_block_lists_equal(expected, result_blocks)
    assert 0 == tokens_remaining

def test_distribute_tokens_not_in_arbitrator_is_last(my_arbitrator):
    blocks = [
        Block(name='system_prompt', data='aaa', to_text=str),
        Block(name='chat_history', data='bbb', to_text=str),
        Block(name='user_input', data='ccc', to_text=str),
        Block(name='actions', data='ddd', to_text=str),
        Block(name='not_in_arbitrators', data='eee', to_text=str),
    ]
    expected = [
        Block(name='system_prompt', data='aaa', to_text=str),
        Block(name='chat_history', data='bbb', to_text=str),
        Block(name='user_input', data='ccc', to_text=str),
        Block(name='actions', data='ddd', to_text=str),
        Block(name='not_in_arbitrators', data='e', to_text=str),
    ]
    tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(list, blocks, 13)
    assert_block_lists_equal(expected, result_blocks)
    assert 0 == tokens_remaining

def test_distribute_tokens_arbitrator_can_be_reused(my_blocks, my_arbitrator):
    for i in range(10):
        tokens_remaining, result_blocks = my_arbitrator.distribute_tokens(my_tokenizer, my_blocks, 2000)
        assert_block_lists_equal(my_blocks, result_blocks)
        assert 0 == tokens_remaining