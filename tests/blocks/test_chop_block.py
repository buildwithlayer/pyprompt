import pytest

from pyprompt.blocks import ChopBlock
from pyprompt.tokenizers import SimpleTokenizer


# ------------------------ Chop Block ------------------------
def test_chop_block():
    block = ChopBlock(name="user_input", data="a b c d", tokenizer=SimpleTokenizer())
    
    assert block.name == "user_input"
    assert block.data == "a b c d"
    assert isinstance(block.tokenizer, SimpleTokenizer)
    assert block.chop_type == 'end'


def test_chop_block_default():
    block = ChopBlock(name="user_input", data="a b c d")
    
    formatted = block.format()
    assert block.data == formatted

    truncated, new_length = block.truncate(3)
    assert truncated == "a b c"
    assert new_length == 3


def test_chop_block_start():
    block = ChopBlock(name="user_input", data="a b c d", tokenizer=SimpleTokenizer())
    
    formatted = block.format()
    assert block.data == formatted

    truncated, new_length = block.truncate(3)
    assert truncated == "a b"
    assert new_length == 3


def test_chop_block_extra():
    block = ChopBlock(name="user_input", data="a b c d")
    
    formatted = block.format()
    assert block.data == formatted

    truncated, new_length = block.truncate(7)
    assert truncated == "a b c d"
    assert new_length == 4
