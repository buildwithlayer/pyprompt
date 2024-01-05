from pyprompt.blocks import Block

# ------------------------ Base Block ------------------------
def test_block_pass():
    block = Block(name="user_input", data="bbb")

    assert block.name == "user_input"
    assert block.data == "bbb"
