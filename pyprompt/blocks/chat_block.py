from pyprompt.blocks import BaseBlock
from math import floor
import logging

logger = logging.getLogger(__name__)

class ChatBlock(BaseBlock):
    def __init__(self, name, truncation_type='simple', max_tokens=1000, **kwargs):

        self.max_tokens = max_tokens
        self.truncation_type = truncation_type
        
        def truncator(_):
            if self.truncation_type == 'simple':
                return ChatBlock._simple_truncate(self._tokenizer, self._data, max_tokens)
        
        super().__init__(name, truncator=truncator, **kwargs)
        self._validate()
    
    @staticmethod
    def _simple_truncate(tokenizer, data, max_tokens):
        messages = []
        tokens = 0
        for message in data:
            message_tokens = tokenizer.tokenize(message)
            if tokens + len(message_tokens) <= max_tokens:
                messages.append(message)
                tokens += len(message_tokens)
            else:
                break
        return messages
        
    
        