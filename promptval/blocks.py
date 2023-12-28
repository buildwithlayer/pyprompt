from math import floor

class Block:
    """
    Represents a section of the prompt, including its name and its contents.
    To create a block with more specific behavior, derive from this class.
    """

    def _validate(self):
        """
        Validates the block.
        """
        assert self.name is not None
        assert self.data is not None
        assert self.to_text is not None

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

    def reduce(self, factor):
        """
        Creates a new block with content reduced by the given factor.
        """
        min_idx = len(self.data) - floor(len(self.data) * factor)
        return Block(self.name, self.data[min_idx:], self.to_text)


    def __str__(self):
        return self.to_text(self.data)
    
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
        assert self.name is not None
        assert self.data is not None
        assert self.to_text is not None

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

    
# For demonstration purposes
example_blocks = [
    Block(name="system_prompt", data="aaa", to_text=str),
    Block(name="user_input", data="bbb", to_text=str),
    Block(name="chat_history", data="ccc", to_text=str),
    Block(name="actions", data="ddd", to_text=str),
]