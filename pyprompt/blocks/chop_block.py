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

        self.max_tokens = max_tokens
        self.chop_block = chop_block
        
        def truncator(_):            
            return ChopBlock._chop(self._tokenizer, self._data, max_tokens, chop_block)
        
        super().__init__(name, truncator=truncator, **kwargs)
        self._validate()
        
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
    
    @staticmethod
    def _chop(tokenizer, data, max_tokens, chop_block):
        """
        Reduces the block's data by chopping off tokens.
        """
        logger.info(f'Chopping block content')
        
        print(f"Chopping block content: {data}", max_tokens, chop_block)
        tokens = tokenizer.encode(data)
        if chop_block == 'start':
            return tokenizer.decode(tokens[-max_tokens:])
        elif chop_block == 'end':
            return tokenizer.decode(tokens[:max_tokens])
        else:
            raise ValueError(f"Invalid chop_block direction: {chop_block}")
        
