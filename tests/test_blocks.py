import pytest
from blocks import Block, StrictBlock

def test_block_pass():
    block = Block(name="user_input", data="bbb", to_text=str)
    assert block.name == "user_input"
    assert block.data == "bbb"
    assert block.to_text == str

def test_block_fail():
    with pytest.raises(AssertionError):
        Block(name=None, data="bbb", to_text=str)

def test_block_reduce():
    block = Block(name="user_input", data="abcde", to_text=str)
    reduced_block = block.reduce(0.5)
    assert reduced_block.name == "user_input"
    assert reduced_block.data == "de"
    assert reduced_block.to_text == str

    # To check that the block is not mutated
    assert block.name == "user_input"
    assert block.data == "abcde"
    assert block.to_text == str

def test_strict_block_pass():
    block = StrictBlock(name="user_input", data="bbb", to_text=str)
    assert block.name == "user_input"
    assert block.data == "bbb"
    assert block.to_text == str

def test_strict_block_fail():
    with pytest.raises(AssertionError):
        StrictBlock(name=None, data="bbb", to_text=str)