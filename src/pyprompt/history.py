from sys import maxsize

from .messages import Message
from .prompt_element import PromptElement

__all__ = ("History",)


class History(PromptElement):
    """
    Manages conversation history with priority-based token pruning.

    Works with:
    - PromptElement: Base class for hierarchical prompt structures
    - Message: For conversation messages (UserMessage, AssistantMessage, etc.)

    Features:
    - Linear or stepped priority scaling
    - Newest-to-oldest message prioritization
    - Automatic token budget management
    """

    def __init__(
        self,
        *children: Message,
        newest_priority: int = maxsize,
        oldest_priority: int | None = None,
        step_priority: int | None = None,
        **kwargs,
    ):
        super().__init__(*children, **kwargs)

        if oldest_priority is None and step_priority is None:
            oldest_priority = 0

        self.newest_priority = newest_priority
        self.oldest_priority = oldest_priority
        self.step_priority = step_priority

        for idx, child in enumerate(self.children):
            child.priority = self.get_child_priority(idx)

    def get_child_priority(self, idx: int) -> int:
        if self.step_priority is None:
            if idx == len(self.children) - 1:
                return self.newest_priority
            return round(
                self.oldest_priority
                + (self.newest_priority - self.oldest_priority)
                * idx
                / (len(self.children) - 1)
            )
        return self.newest_priority - self.step_priority * (
            len(self.children) - 1 - idx
        )
