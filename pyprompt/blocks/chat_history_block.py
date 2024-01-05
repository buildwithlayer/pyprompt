from typing import Callable, List, Optional, Tuple, TypedDict
from .block import Block
from enum import Enum

__all__ = ["ChatHistoryBlock", "Message", "Role"]

class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: str


class ChatHistoryBlock(Block[List[Message]]):
    """Represents a block that displays chat history."""
    
    def format(self, data: Optional[List[Message]] = None) -> str:
        """Formats the chat history data into a string representation."""
        
        if data is None:
            data = self.data
        
        formatted_messages = [
            f"{message['role']}: {message['content']}" for message in data
        ]
        
        return "\n".join(formatted_messages)
    
    def truncate(
            self,
            max_tokens: int,
            truncation_type: Optional[str] = "simple",
            truncation_method: Optional[Callable[[int], Tuple[List[Message], int]]] = None,
    ) -> Tuple[List[Message], int]:
        """Truncates the chat history data to fit within a maximum number of tokens.
        
        Args:
            max_tokens (int): The maximum number of tokens allowed.
            truncation_type (Optional[str]): The type of truncation to perform. Defaults to "simple".
            truncation_method: Override the truncation method. Defaults to None.
        
        Returns:
            Tuple[List[Message], int]: The truncated chat history data and the number of tokens used.
        """
        if truncation_method is None:
            if truncation_type == "simple":
                truncation_method = self._simple_truncate
            elif truncation_type == "summary":
                truncation_method = self._summary_truncate
            else:
                raise ValueError(f"Unknown truncation type: {truncation_type}")
            
        return truncation_method(max_tokens)
        

    def _simple_truncate(self, max_tokens: int):
        """Fits as many full messages as possible into the given number of tokens."""
        for i in range(0, len(self.data)):
            new_data = self.data[i:]
            formatted = self.format(new_data)
            encoded = self.tokenizer.encode(formatted)
            if len(encoded) <= max_tokens:
                return new_data, len(encoded)
        
    def _summary_truncate(self, max_tokens: int):
        """Truncates the chat history data using a summary truncation method."""
        raise NotImplementedError
