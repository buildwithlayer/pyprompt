from pyprompt.tokenizers import SimpleTokenizer

# ------------------------ Simple Tokenizer ------------------------
def test_simple_tokenizer():
    encoded = SimpleTokenizer.encode("a b c d!")
    
    assert encoded == ['a', 'b', 'c', 'd!']
    
    decoded = SimpleTokenizer.decode(['a', 'b', 'c', 'd!'])
    
    assert decoded == "a b c d!"