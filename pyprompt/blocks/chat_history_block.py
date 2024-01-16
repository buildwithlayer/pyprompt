from __future__ import annotations
from typing import Callable, List, Optional, Tuple, TypedDict, Union

from .block import Block, Buildables
from enum import Enum

__all__ = ["ChatHistoryBlock", "Message", "Role", "MessageTruncator"]

class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: str
    
MessageTruncator = Callable[[int], Tuple[List[Message], int]]


class ChatHistoryBlock(Block[List[Message]]):
    """Represents a block that displays chat history."""
    
    class Formats(Enum):
        MESSAGES = "messages"
        STRING = "string"
    
    @staticmethod
    def _string_format(data: List[Message]):
        formatted_messages = [
            f"{message['role']}: {message['content']}" for message in data
        ]
        
        return "\n".join(formatted_messages)
    
    def format(self, to: ChatHistoryBlock.Formats = None, **kwargs) -> ChatHistoryBlock:
        """Formats the chat history data into a string representation."""
        
        if to == None:
            to = ChatHistoryBlock.Formats.STRING
        
        if to == ChatHistoryBlock.Formats.STRING:
            self.data = ChatHistoryBlock._string_format(self.data)
            
        return self
    
    def truncate(
            self,
            max_tokens: int,
            truncate: Optional[Union[str, MessageTruncator]] = "simple",
    ) -> ChatHistoryBlock:
        
        if callable(truncate):
            self.data, _ = truncate(max_tokens, self.data)
        elif truncate == "simple":
            self.data, _ = self._simple_truncate(max_tokens)
        elif truncate == "summary":
            self.data, _ = self._summary_truncate(max_tokens)
        else:
            raise ValueError(f"Unknown truncation type: {truncate}")
        
        return self
    
    def _count_messages_tokens(self, messages: List[Message]):
        stringified = [f"{message['role']}: {message['content']}" for message in messages]
        encoded = [self.tokenizer.encode(message) for message in stringified]
        sumed = sum([len(tokens) for tokens in encoded])
        return sumed

    def _simple_truncate(self, max_tokens: int) -> Tuple[List[Message], int]:
        total_tokens = 0

        messages = self.data.copy()
        
        count = self._count_messages_tokens(messages)
        
        while self._count_messages_tokens(messages) > max_tokens:
            messages.pop(0)
            
        total_tokens = self._count_messages_tokens(messages)

        if len(messages) == 1:
            pass
        
        return messages, total_tokens
        
    def _summary_truncate(self, max_tokens: int) -> Tuple[List[Message], int]:
        """Truncates the chat history data using a summary truncation method."""
        raise NotImplementedError