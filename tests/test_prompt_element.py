from sys import maxsize

from conftest import decode, encode

from pyprompt import *


def test_priority_chain():
    prompt = PromptElement(
        PromptElement("It is 27 degrees today.", pass_priority=True),
        "Thank you!",
        priority=200,
    )
    assert prompt.priority_chain == [200, maxsize]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", priority=100),
        PromptElement(
            PromptElement("It is 27 degrees today.", pass_priority=True),
            "Thank you!",
            priority=200,
        ),
    )
    assert prompt.priority_chain == [maxsize, maxsize]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", priority=100),
        PromptElement(
            PromptElement("It is 27 degrees today.", pass_priority=True),
            "Thank you!",
            priority=200,
        ),
        PromptElement(
            "Foo",
            PromptElement(
                "Bar",
                PromptElement("FooBar", priority=20),
                priority=10,
            ),
            PromptElement("Another FooBar", pass_priority=True),
            priority=1,
        ),
    )
    assert prompt.priority_chain == [maxsize, maxsize]


def test_priorities():
    prompt = PromptElement(
        PromptElement("It is 27 degrees today.", pass_priority=True),
        "Thank you!",
        priority=200,
    )
    assert prompt.priorities == [
        ([maxsize], [1]),
        ([maxsize], [0, 0]),
    ]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", priority=100),
        PromptElement(
            PromptElement("It is 27 degrees today.", pass_priority=True),
            "Thank you!",
            priority=200,
        ),
    )
    assert prompt.priorities == [
        ([maxsize], [0]),
        ([200, maxsize], [2]),
        ([100, maxsize], [1]),
    ]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", priority=100),
        PromptElement(
            PromptElement("It is 27 degrees today.", pass_priority=True),
            "Thank you!",
            priority=200,
        ),
        PromptElement(
            "Foo",
            PromptElement(
                "Bar",
                PromptElement("FooBar", priority=20),
                priority=10,
            ),
            PromptElement("Another FooBar", pass_priority=True),
            priority=1,
        ),
    )
    assert prompt.priorities == [
        ([maxsize], [0]),
        ([200, maxsize], [2]),
        ([100, maxsize], [1]),
        ([1, maxsize], [3]),
    ]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", priority=100, reserve=6),
        PromptElement(
            PromptElement("It is 27 degrees today.", priority=200),
            "Thank you!",
            pass_priority=True,
        ),
    )
    assert prompt.priorities == [
        ([maxsize], [2, 1]),
        ([maxsize], [0]),
        ([200, maxsize], [2, 0]),
        ([100, maxsize], [1]),
    ]


def test_get_token_map():
    prompt = PromptElement(
        "Hello!",
        PromptElement(f"How are you, Bob?"),
        PromptElement(
            PromptElement(f"It is 27 degrees today."),
            "Thank you!",
        ),
    )
    assert prompt.get_token_map(encode) == {
        0: [9906, 0],
        1: {
            0: [4438, 527, 499, 11, 14596, 30],
        },
        2: {
            0: {
                0: [2181, 374, 220, 1544, 12628, 3432, 13],
            },
            1: [13359, 499, 0],
        },
    }


def test_get_reserved():
    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?"),
        PromptElement(
            PromptElement("It is 27 degrees today."),
            "Thank you!",
        ),
    )
    budget = 100
    assert prompt.get_reserved(budget) == 0

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", reserve=10),
        PromptElement(
            PromptElement("It is 27 degrees today.", reserve=1 / 5),
            "Thank you!",
        ),
    )
    budget = 100
    assert prompt.get_reserved(budget) == 30


def test_get_token_count():
    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?"),
        PromptElement(
            PromptElement("It is 27 degrees today."),
            "Thank you!",
        ),
    )
    token_map = {
        0: [13225, 0],
        1: {
            0: [5299, 553, 481, 11, 22582, 30],
        },
        2: {
            0: {
                0: [3206, 382, 220, 2092, 18210, 4044, 13],
            },
            1: [13659, 481, 0],
        },
    }
    assert prompt.get_token_count(token_map) == 18


def test_prune_no_priority():
    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?"),
        PromptElement(
            PromptElement("It is 27 degrees today."),
            "Thank you!",
        ),
    )
    token_map = {
        0: [13225, 0],
        1: {
            0: [5299, 553, 481, 11, 22582, 30],
        },
        2: {
            0: {
                0: [3206, 382, 220, 2092, 18210, 4044, 13],
            },
            1: [13659, 481, 0],
        },
    }

    token_map = prompt.prune(18, token_map, encode, decode)
    assert token_map == {
        0: [13225, 0],
        1: {
            0: [5299, 553, 481, 11, 22582, 30],
        },
        2: {
            0: {
                0: [3206, 382, 220, 2092, 18210, 4044, 13],
            },
            1: [13659, 481, 0],
        },
    }

    token_map = prompt.prune(17, token_map, encode, decode)
    assert token_map == {
        1: {
            0: [5299, 553, 481, 11, 22582, 30],
        },
        2: {
            0: {
                0: [3206, 382, 220, 2092, 18210, 4044, 13],
            },
            1: [13659, 481, 0],
        },
    }

    token_map = prompt.prune(12, token_map, encode, decode)
    assert token_map == {
        2: {
            0: {
                0: [3206, 382, 220, 2092, 18210, 4044, 13],
            },
            1: [13659, 481, 0],
        },
    }


def test_prune_with_priority():
    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", priority=100),
        PromptElement(
            PromptElement("It is 27 degrees today.", priority=200),
            "Thank you!",
            pass_priority=True,
        ),
    )
    token_map = {
        0: [13225, 0],  # maxsize
        1: {
            0: [5299, 553, 481, 11, 22582, 30],  # 100
        },
        2: {
            0: {  # 200
                0: [3206, 382, 220, 2092, 18210, 4044, 13],  # maxsize
            },
            1: [13659, 481, 0],  # maxsize
        },
    }

    token_map = prompt.prune(17, token_map, encode, decode)
    assert token_map == {
        0: [13225, 0],  # maxsize
        2: {
            0: {  # 200
                0: [3206, 382, 220, 2092, 18210, 4044, 13],  # maxsize
            },
            1: [13659, 481, 0],  # maxsize
        },
    }

    token_map = prompt.prune(12, token_map, encode, decode)
    assert token_map == {
        0: [13225, 0],  # maxsize
        2: {
            0: {  # 200
                0: [3206, 382, 220, 2092, 18210, 4044, 13],  # maxsize
            },
            1: [13659, 481, 0],  # maxsize
        },
    }

    token_map = prompt.prune(10, token_map, encode, decode)
    assert token_map == {
        0: [13225, 0],  # maxsize
        2: {
            1: [13659, 481, 0],  # maxsize
        },
    }


def test_prune_with_reserve():
    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, Bob?", priority=100, reserve=6),
        PromptElement(
            PromptElement("It is 27 degrees today.", priority=200),
            "Thank you!",
            pass_priority=True,
        ),
    )
    token_map = {
        0: [13225, 0],  # maxsize
        1: {
            0: [5299, 553, 481, 11, 22582, 30],  # 100
        },
        2: {
            0: {  # 200
                0: [3206, 382, 220, 2092, 18210, 4044, 13],  # maxsize
            },
            1: [13659, 481, 0],  # maxsize
        },
    }

    token_map = prompt.prune(18, token_map, encode, decode)
    assert token_map == {
        0: [13225, 0],  # maxsize
        1: {
            0: [5299, 553, 481, 11, 22582, 30],  # 100
        },
        2: {
            0: {  # 200
                0: [3206, 382, 220, 2092, 18210, 4044, 13],  # maxsize
            },
            1: [13659, 481, 0],  # maxsize
        },
    }

    token_map = prompt.prune(17, token_map, encode, decode)
    assert token_map == {
        0: [13225, 0],  # maxsize
        1: {
            0: [5299, 553, 481, 11, 22582, 30],  # 100
        },
        2: {
            1: [13659, 481, 0],  # maxsize
        },
    }

    token_map = prompt.prune(10, token_map, encode, decode)
    assert token_map == {
        1: {
            0: [5299, 553, 481, 11, 22582, 30],  # 100
        },
        2: {
            1: [13659, 481, 0],  # maxsize
        },
    }

    token_map = prompt.prune(5, token_map, encode, decode)
    assert token_map is None


def test_grow():
    def grow_cb(element, encoding_func, decoding_func, grow_state):
        if grow_state is None:
            grow_state = 2
        else:
            grow_state += 1

        parts = []
        for i in range(1, grow_state + 1):
            parts.append("Hello" + "!" * i)
        content = " ".join(parts)

        return {0: encoding_func(content)}, grow_state

    prompt = PromptElement(
        "Hello!",
        grow_callback=grow_cb,
    )
    token_map, final_grow_state = prompt.grow(
        20,
        prompt.get_token_map(encoding_func=encode),
        encode,
        decode,
    )
    assert token_map == {
        0: [
            9906,
            0,
            22691,
            3001,
            22691,
            12340,
            22691,
            17523,
            22691,
            70900,
            22691,
            17523,
            3001,
            22691,
            17523,
            12340,
            22691,
            51767,
            22691,
            17523,
            70900,
        ],
    }
    assert final_grow_state == 9
