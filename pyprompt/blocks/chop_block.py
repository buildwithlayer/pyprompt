from pyprompt.blocks import BaseBlock
from math import floor
import logging

logger = logging.getLogger(__name__)

class ChopBlock(BaseBlock):
    """
    A block that reduces its data by chopping off tokens.

    Args:
        name (str): The name of the block.
        
        **kwargs: Additional keyword arguments.
            max_tokens (int): The maximum number of tokens to keep. Default 1000.
            chop_block (str): The direction to chop_block off tokens. Can be 'start' or 'end'.
    """

    def __init__(self, name, chop_block='start', max_tokens=1000, **kwargs):
        super().__init__(name, **kwargs)
        self.max_tokens = max_tokens
        
        print(f"chopdirection: {chop_block}")
        self.chop_block = chop_block
        
    def _validate(self):
        """
        Validates the block.
        """
        super()._validate()
        if not isinstance(self.max_tokens, int):
            raise TypeError(f"max_tokens must be an int, not {type(self.max_tokens)}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative, not {self.max_tokens}")
        if self.chop_block not in ['start', 'end']:
            raise ValueError(f"Invalid chop_block direction: {self.chop_block}")

    def truncate(self):
        """
        Reduces the block's data by chopping off tokens.
        """
        logger.info(f'Chopping block content')
        tokens = self._tokenizer.encode(self._data)
        
        if self.chop_block == 'start':
            self._data = self._tokenizer.decode(tokens[-self.max_tokens:])
        elif self.chop_block == 'end':
            self._data = self._tokenizer.decode(tokens[:self.max_tokens])
        else:
            raise ValueError(f"Invalid chop_block direction: {self.chop_block}")
        
    def populate(self, data):
        
        self._data = data
        self.truncate()
    
