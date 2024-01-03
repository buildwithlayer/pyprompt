from math import floor
import logging

logger = logging.getLogger(__name__)

class BaseBlock:
    
    def __init__(self, name, **kwargs):
            """
            Initializes this block object.

            Args:
                name (str): The name of the block.
                **kwargs: Additional keyword arguments.
                    data (Any): Optional data for the block.

            """
            self.name = name
            self._data = kwargs.get('data')
            
            self._validate()
    
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
        
    def apply(self, func):
        """
        Applies the given function to the block's data.
        """
        logger.info(f'Running reduce on {self.name} with function {func}.')
        
        self._data = func(self._data)