import pytest

from pyprompt import *


@pytest.fixture(scope="function")
def sample_token_map() -> TokenMap:
    return {
        0: [0],
        1: [1, 2, 3],
        2: {
            0: [4],
        },
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }


def test_orphan_child_from_token_map(sample_token_map: TokenMap):
    parent_map, child_map = orphan_child_from_token_map(sample_token_map, [0])
    assert parent_map == {
        1: [1, 2, 3],
        2: {
            0: [4],
        },
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }
    assert child_map == [0]

    parent_map, child_map = orphan_child_from_token_map(parent_map, [2, 0])
    assert parent_map == {
        1: [1, 2, 3],
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }
    assert child_map == [4]

    parent_map, child_map = orphan_child_from_token_map(parent_map, [3, 1])
    assert parent_map == {
        1: [1, 2, 3],
        3: {
            0: [5],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }
    assert child_map == [6, 7, 8]

    parent_map, child_map = orphan_child_from_token_map(parent_map, [3, 4, 0])
    assert parent_map == {
        1: [1, 2, 3],
        3: {
            0: [5],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
        }
    }
    assert child_map == {
        0: [14, 15, 16],
    }


def test_update_token_map():
    token_map = {
        1: [1, 2, 3],
        3: {
            0: [5],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
        }
    }

    descendant_indices = [3, 4, 0]
    value = {
        0: [14, 15, 16],
    }
    token_map = update_token_map(token_map, descendant_indices, value)
    assert token_map == {
        1: [1, 2, 3],
        3: {
            0: [5],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    descendant_indices = [3, 1]
    value = [6, 7, 8]
    token_map = update_token_map(token_map, descendant_indices, value)
    assert token_map == {
        1: [1, 2, 3],
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    descendant_indices = [2, 0]
    value = [4]
    token_map = update_token_map(token_map, descendant_indices, value)
    assert token_map == {
        1: [1, 2, 3],
        2: {
            0: [4],
        },
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    descendant_indices = [0]
    value = [0]
    token_map = update_token_map(token_map, descendant_indices, value)
    assert token_map == {
        0: [0],
        1: [1, 2, 3],
        2: {
            0: [4],
        },
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }


def test_prune_from_token_map(sample_token_map: TokenMap):
    token_map = update_token_map(sample_token_map, [0], None)
    assert token_map == {
        1: [1, 2, 3],
        2: {
            0: [4],
        },
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    token_map = update_token_map(token_map, [1], None)
    assert token_map == {
        2: {
            0: [4],
        },
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    token_map = update_token_map(token_map, [2, 0], None)
    assert token_map == {
        3: {
            0: [5],
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    token_map = update_token_map(token_map, [3, 0], None)
    assert token_map == {
        3: {
            1: [6, 7, 8],
            2: {
                0: [9],
                1: [10, 11, 12],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    token_map = update_token_map(token_map, [3, 2, 1], None)
    assert token_map == {
        3: {
            1: [6, 7, 8],
            2: {
                0: [9],
            },
            3: [13],
            4: {
                0: {
                    0: [14, 15, 16],
                }
            }
        }
    }

    token_map = update_token_map(token_map, [3], None)
    assert token_map is None
