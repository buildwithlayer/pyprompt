from sys import maxsize

from conftest import encode, decode
from pyprompt import *


def test_priority_chain():
    prompt = PromptElement(
        PromptElement("It is {temperature} degrees today.", pass_priority=True),
        "Thank you!",
        priority=200,
    )
    assert prompt.priority_chain == [200, maxsize]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, {name}?", priority=100),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", pass_priority=True),
            "Thank you!",
            priority=200,
        ),
    )
    assert prompt.priority_chain == [maxsize, maxsize]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, {name}?", priority=100),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", pass_priority=True),
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
        PromptElement("It is {temperature} degrees today.", pass_priority=True),
        "Thank you!",
        priority=200,
    )
    assert prompt.priorities == [
        ([maxsize], [1]),
        ([maxsize], [0, 0]),
    ]

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, {name}?", priority=100),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", pass_priority=True),
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
        PromptElement("How are you, {name}?", priority=100),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", pass_priority=True),
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
        PromptElement("How are you, {name}?", priority=100, reserve=6),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", priority=200),
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
        PromptElement("How are you, {name}?"),
        PromptElement(
            PromptElement("It is {temperature} degrees today."),
            "Thank you!",
        ),
    )
    props = {
        "name": "Bob",
        "temperature": 27,
    }
    assert prompt.get_token_map(props, encode) == {
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
        PromptElement("How are you, {name}?"),
        PromptElement(
            PromptElement("It is {temperature} degrees today."),
            "Thank you!",
        ),
    )
    budget = 100
    assert prompt.get_reserved(budget) == 0

    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, {name}?", reserve=10),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", reserve=1/5),
            "Thank you!",
        ),
    )
    budget = 100
    assert prompt.get_reserved(budget) == 30


def test_get_token_count():
    prompt = PromptElement(
        "Hello!",
        PromptElement("How are you, {name}?"),
        PromptElement(
            PromptElement("It is {temperature} degrees today."),
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
        PromptElement("How are you, {name}?"),
        PromptElement(
            PromptElement("It is {temperature} degrees today."),
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
        PromptElement("How are you, {name}?", priority=100),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", priority=200),
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
        PromptElement("How are you, {name}?", priority=100, reserve=6),
        PromptElement(
            PromptElement("It is {temperature} degrees today.", priority=200),
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
