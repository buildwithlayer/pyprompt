from pyprompt.blocks import ChatHistoryBlock
from pyprompt.tokenizers import SimpleTokenizer


# ------------------------ ChatBlock ------------------------

sample_message_data = [
    {"role": "system", "content": "Do you think you are cool"},
    {"role": "user", "content": "like really cool?"},
    {"role": "user", "content": "no I not :("},
]    

def test_chat_block():
    block = ChatHistoryBlock(name="user_input", data=sample_message_data, tokenizer=SimpleTokenizer())
    
    assert block.data == sample_message_data

    truncated, length = block.truncate(4)
    assert truncated == []
    assert length == 0
    
    # test 1 message
    truncated, length = block.truncate(5)
    assert truncated == sample_message_data[2:]
    assert length == 5
    
    # test 1 extra space
    truncated, length = block.truncate(7)
    assert truncated == sample_message_data[2:]
    assert length == 5

    # test all messages
    truncated, length = block.truncate(1000)
    assert truncated == sample_message_data
    assert length == 16

    truncated, length = block.truncate(9)
    assert truncated == sample_message_data[1:]
    assert length == 9



