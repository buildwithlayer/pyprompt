import logging
import tiktoken

logger = logging.getLogger(__name__)

class BaseBlock:
    
    def __init__(self, name, **kwargs):
            """
            Initializes this block object.

            Args:
                name (str): The name of the block.
                **kwargs: Additional keyword arguments.
                    data (Any): Optional data for the block.
                    tokenizer (Callable): Optional tokenizer for the block. Defaults to the cl100k_base tokenizer. 
                    Tokenizers must have an encode and decode method.

            """
            self.name = name
            self._data = kwargs.get('data')
            self._tokenizer = kwargs.get('tokenizer') or tiktoken.get_encoding("cl100k_base")

    
    def _validate(self):
        """
        Validates the block.
        """
        if not isinstance(self.name, str):
            raise TypeError(f'Block name must be a string, not {type(self.name)}.')
        
    def apply(self, func):
        """
        Applies the given function to the block's data.
        """
        logger.info(f'Running reduce on {self.name} with function {func}.')
        
        self._data = func(self._data)
        
    def truncate(self, func=None):
        """
        Truncates the block's data if a function is given, otherwise raises a NotImplementedError.
        """
        if func is not None:
            self.apply(func)
        else:
            raise NotImplementedError(f"{__file__} This method must be implemented by a derived class.")
            
    def populate(self, data, **kwargs):
        """
        Attempts to populate the block with additional tokens.
        """
        self._data = data
