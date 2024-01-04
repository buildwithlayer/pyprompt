import pytest
from pyprompt.blocks import Block, StrictBlock, ChopBlock, BaseBlock

# ------------------------ Helpers ------------------------
def populator(block):
    return "ccc"

def addative_populator(block):
    return "ccc " + block._data 

# ------------------------ Base Block ------------------------
def test_base_block_pass():
    block = BaseBlock(name="user_input", data="bbb")
    assert block.name == "user_input"
    assert block._data == "bbb"
    
def test_base_block_fail():
    with pytest.raises(TypeError):
        BaseBlock(name=None, data="bbb")
    
def test_base_block_populator():
    block = BaseBlock(name="user_input", populator=populator)
        
    assert block.name == "user_input"
    assert block._data == "ccc"
    assert block._populator == populator
    
def test_base_block_populator_and_data():
    block = BaseBlock(name="user_input", data="bbb", populator=populator)
    
    assert block.name == "user_input"
    assert block._data == "bbb"
    assert block._populator == populator
    
    block.repopulate()
    assert block.name == "user_input"
    assert block._data == "ccc"
    
# ------------------------ Chop Block ------------------------
def test_chop_block_start():
    block = ChopBlock(name="user_input", chop_block='start', max_tokens=3, data="a b c d")
    
    assert block.name == "user_input"
    assert block.max_tokens == 3
    assert block.chop_block == 'start'
    assert block._data == " b c d"  # not sure why there's a space here
    
def test_chop_block_end():
    block = ChopBlock(name="user_input", chop_block='end', max_tokens=3, data="a b c d")
    
    assert block.name == "user_input"
    assert block.max_tokens == 3
    assert block.chop_block == 'end'
    assert block._data == "a b c"
    
def test_chop_block_repopulate():
    block = ChopBlock(name="user_input", chop_block='end', max_tokens=3, data="a b c d", populator=addative_populator)
    
    assert block.name == "user_input"
    assert block.max_tokens == 3
    assert block.chop_block == 'end'
    assert block._data == "a b c"
    
    block.repopulate()
    assert block.name == "user_input"
    assert block._data == "ccc a b"
    
def test_chop_block_fail():
    with pytest.raises(ValueError):
        ChopBlock(name="user_input", chop_block='middle', max_tokens=3, data="a b c d")
        
    with pytest.raises(TypeError):
        ChopBlock(name="user_input", chop_block='start', max_tokens="3", data="a b c d")
        
    with pytest.raises(ValueError):
        ChopBlock(name="user_input", chop_block='start', max_tokens=-1, data="a b c d")
        
