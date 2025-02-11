from conftest import decode, encode

from pyprompt import *


def test_prune():
    prompt = TextChunk("Hello, world!")
    token_map = {
        0: [13225, 11, 2375, 0],
    }

    token_map = prompt.prune(3, token_map, encode, decode)
    assert token_map == {
        0: [13225, 11, 2375],
    }

    token_map = prompt.prune(2, token_map, encode, decode)
    assert token_map == {
        0: [13225, 11],
    }

    token_map = prompt.prune(0, token_map, encode, decode)
    assert token_map is None
