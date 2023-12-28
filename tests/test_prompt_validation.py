from test_utilities import *
from prompt_validation import Prompt

# Test distribute tokens

def test_prompt_correct_tokens(my_blocks, my_arbitrator):
    prompt = Prompt(
        blocks=my_blocks,
        arbitrator=my_arbitrator,
        tokenize=my_tokenizer,
        max_tokens=2000,
    )
    print('heere')
    assert_block_lists_equal(my_blocks, prompt.blocks)
    assert 0 == prompt.max_tokens - prompt.tokens_used

def test_prompt_excess_tokens(my_blocks, my_arbitrator):
    prompt = Prompt(
        blocks=my_blocks,
        arbitrator=my_arbitrator,
        tokenize=my_tokenizer,
        max_tokens=2077,
    )
    assert_block_lists_equal(my_blocks, prompt.blocks)
    assert 77 == prompt.max_tokens - prompt.tokens_used

def test_prompt_short_tokens(my_blocks, my_arbitrator):
    prompt = Prompt(
        blocks=my_blocks,
        arbitrator=my_arbitrator,
        tokenize=my_tokenizer,
        max_tokens=499,
    )
    expected = make_blocks_helper(('a'*998, str), ('', str), ([], lambda lst: ''.join(str(x) for x in lst)), ('', str))
    assert_block_lists_equal(expected, prompt.blocks)
    assert 0 == prompt.max_tokens - prompt.tokens_used