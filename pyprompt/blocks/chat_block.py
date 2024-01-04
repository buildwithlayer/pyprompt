from pyprompt.blocks import BaseBlock
from math import floor
import logging

logger = logging.getLogger(__name__)

class ChatBlock(BaseBlock):
    """
    A block that represents a chat conversation.

    Args:
        name (str): The name of the block.
        data (list): The list of chat messages.
        truncation_type (str, optional): The type of truncation to apply. Defaults to 'simple'.
        max_tokens (int, optional): The maximum number of tokens allowed in the block. Defaults to 1000.
        
        **kwargs: Additional keyword arguments to be passed to the base class.
    """
    def __init__(self, name, *, data, truncation_type='simple', max_tokens=1000, **kwargs):
        self.max_tokens = max_tokens
        self.truncation_type = truncation_type
        
        def truncator(_):
            if self.truncation_type == 'simple':
                return ChatBlock._simple_truncate(self._tokenizer, self._data, max_tokens)
        
        super().__init__(name, data=data, truncator=truncator, **kwargs)
        self._validate()
    
    @staticmethod
    def _simple_truncate(tokenizer, data, max_tokens):
        """
        Truncates the chat messages based on the maximum number of tokens.

        Args:
            tokenizer: The tokenizer used to tokenize the messages.
            data (list): The list of chat messages.
            max_tokens (int): The maximum number of tokens allowed in the block.

        Returns:
            list: The truncated chat messages.

        """
        messages = []
        tokens = 0
        
        while tokens <= max_tokens:
            message = data.pop()
            message_tokens = tokenizer.encode(message["role"]) + tokenizer.encode(message["content"])
            
            if tokens + len(message_tokens) <= max_tokens:
                messages = [message] + messages
                tokens += len(message_tokens)
            else:
                break
            
        return messages
        
    
        