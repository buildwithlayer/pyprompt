from pyprompt.blocks import BaseBlock
from math import floor
import logging

logger = logging.getLogger(__name__)

class ChopBlock(BaseBlock):
    """
    A block that reduces its data by chopping off tokens.

    Args:
        name (str): The name of the block.
        max_tokens (int): The maximum number of tokens to keep.
        
        **kwargs: Additional keyword arguments.
            chop (str): The direction to chop off tokens. Can be 'start' or 'end'.
    """

    def __init__(self, name, max_tokens, chop='start', **kwargs):
        super().__init__(name, **kwargs)
        self.max_tokens = max_tokens
        self.chop = chop

    def reduce(self):
        """
        Reduces the block's data by chopping off tokens.
        """
        logger.info(f'Chopping block content')
        if self.direction == 'start':
            self.data = self.data[:self.max_tokens]
        elif self.direction == 'end':
            self.data = self.data[-self.max_tokens:]
        else:
            raise ValueError(f"Invalid direction: {self.direction}")
    
