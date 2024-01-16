from typing import TypeVar
from .block import Block

__all__ = ["DeconstructableBlock"]

T = TypeVar("T")

class DeconstructableBlock(Block[T]):
    """
    Represents a destructible block, which is a wrapper around Block that indicates it is destructible.
    """
    pass