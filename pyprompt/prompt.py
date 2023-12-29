import tiktoken
import logging
from token_distribution import example_arbitrator
from pyprompt.block import example_blocks

class Prompt:
    """
    Represents a prompt, including its blocks and the arbitrator used to
    distribute tokens among them.
    """

    def _validate(self):
        """Validates the prompt."""
        if self.blocks is None:
            raise TypeError("Blocks cannot be None.")
        if self.arbitrator is None:
            raise TypeError("Arbitrator cannot be None.")

    def _distribute_tokens(self):
        """Distributes tokens amongst the blocks in this prompt."""
        tokens_remaining, self.blocks = self.arbitrator.distribute_tokens(
            tokenize=self.tokenize,
            blocks=self.blocks,
            max_tokens=self.max_tokens,
        )

        self.tokens_used = self.max_tokens - tokens_remaining

    def __init__(self, blocks, arbitrator, tokenize, max_tokens):
        """
        Initializes this prompt object.

        Args:
            blocks: The blocks in this prompt.
            arbitrator: The arbitrator used to distribute tokens among the blocks.
        """
        self.blocks = blocks
        self.arbitrator = arbitrator
        self.max_tokens = max_tokens
        self.tokenize = tokenize

        self._distribute_tokens()
        self._validate()

    def __str__(self):
        """Returns a string representation of this prompt."""
        return '\n'.join([str(block) for block in self.blocks])

def example():
    """
    An example demonstrating how to use this module.
    """
    encoding_name = "cl100k_base"

    tokenize = tiktoken.get_encoding(encoding_name).encode

    prompt = Prompt(
        blocks=example_blocks,
        arbitrator=example_arbitrator,
        tokenize=tokenize,
        max_tokens=10,
    )

    print(prompt)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example()
