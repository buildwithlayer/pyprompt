from math import floor
import logging

logger = logging.getLogger(__name__)
    
class StrictBlock:
    """
    Represents a section of the prompt, including its name and its contents.
    This type of block cannot be reduced.
    To create a block with more specific behavior, derive from this class.
    """

    def _validate(self):
        """
        Validates the block.
        """
        if not isinstance(self.name, str):
            raise TypeError(f'Block name must be a string, not {type(self.name)}.')
        if self.data is None:
            raise TypeError(f'Block data cannot be None.')
        if not callable(self.to_text):
            raise TypeError(f'Block to_text must be a callable, not {type(self.to_text)}.')
        
    def __init__(self, name, data, to_text):
        """
        Initializes this block object.

        Args:
            name: The name of the block.
            data: The contents of the block.
            to_text: The text representation of the data. If the data is
            conceptually empty, an empty string should be returned.
        """
        self.name = name
        self.data = data
        self.to_text = to_text
        
        self._validate()
    
    def __str__(self):
        return self.to_text(self.data)

class Block(StrictBlock):
    """
    Represents a section of the prompt, including its name and its contents.
    To create a block with more specific behavior, derive from this class.
    """

    def reduce(self, factor):
        """
        Creates a new block with content reduced by the given factor.
        """
        logger.info(f'Running reduce on {self.name} with factor {factor}.')
        min_idx = len(self.data) - floor(len(self.data) * factor)
        return Block(self.name, self.data[min_idx:], self.to_text)
    
# For demonstration purposes
example_blocks = [
    Block(name="system_prompt", data="aaa", to_text=str),
    Block(name="user_input", data="bbb", to_text=str),
    Block(name="chat_history", data="ccc", to_text=str),
    Block(name="actions", data="ddd", to_text=str),
]