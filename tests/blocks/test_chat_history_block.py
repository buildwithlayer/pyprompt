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

    truncated, length = block.truncate(17)
    assert truncated == sample_message_data[2:]
    assert length == 17
    
    truncated, length = block.truncate(20)
    assert truncated == sample_message_data[2:]
    assert length == 17

    truncated, length = block.truncate(1000)
    assert truncated == sample_message_data
    assert length == 75

    truncated, length = block.truncate(74)
    assert truncated == sample_message_data[1:]



