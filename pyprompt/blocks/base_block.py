import logging
import tiktoken

logger = logging.getLogger(__name__)

class BaseBlock:
    
    def __init__(self, name, *, data=None, populator=None, tokenizer=None, truncator=None, **kwargs):
        """
        Initializes a BaseBlock object. Either data or populator must be provided. If both are provided, 
        data is used during initialization and populator is used during repopulation.

        Args:
            name (str): The name of the block.
            
            data (Any, optional): Data for the block.
            
            populator (Callable, optional): Called if block needs to be populated. The populator recieves the block as an argument.
            
            tokenizer (Callable, optional): Optional tokenizer for the block. Defaults to the cl100k_base tokenizer.
                Tokenizers must have an encode and decode method.
            
            truncator (Callable, optional): Optional truncator for the block. Defaults to None.
                Truncators must have a truncate method.
            
            **kwargs: Additional keyword arguments.
        """
        if data is None and populator is None:
            raise ValueError("Either 'data' or 'populator' must be provided.")
        
        self.name = name
        self._populator = populator
        self._tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self._truncator = truncator
        
        if data is None:
            self._populate() # Runs the populator function
        else:
            self._data = data
        
        # Apply truncator if provided
        self._data = self._truncator(self) if self._truncator is not None else self._data

        self._validate()

    
    def _validate(self):
        """
        Validates the block.
        """
        if not isinstance(self.name, str):
            raise TypeError(f'Block name must be a string, not {type(self.name)}.')
        
    def _populate(self):
        """
        Repopulates the block content.

        Returns:
            dict: The repopulated block content.
        """
        logger.info(f'_repopulating block content.')

        if self._populator is not None:
            self._data = self._populator(self)
            
        if self._truncator is not None:
            self._data = self._truncator(self)
            
        if self._data is None:
            raise ValueError(f'Block {self.name} has no data.')

        return self._data
        
    def repopulate(self):
        """
        Repopulates the block with data using the populator and truncator functions if provided.
        """
        logger.info(f'Repopulating block content.')
        
        self._data = self._populate()
            
