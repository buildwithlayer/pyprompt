import tiktoken
from token_distribution import example_arbitrator
from blocks import example_blocks

class Prompt:
    """
    Represents a prompt, including its blocks and the arbitrator used to
    distribute tokens among them.
    """

    def _validate(self):
        """Validates the prompt."""
        assert self.blocks is not None
        assert self.arbitrator is not None

    def _distribute_tokens(self):
        """Distributes tokens amongst the blocks in this prompt."""
        tokens_remaining, self.blocks = self.arbitrator.distribute_tokens(
            tokenize=self.tokenize,
            blocks=self.blocks,
            max_tokens=self.max_tokens,
        )

        self.tokens_used = self.max_tokens - tokens_remaining

        # TODO: turn the below into log statements
        print(f"There are {tokens_remaining} tokens remaining.")
        for block in self.blocks:
            print(str(block))

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
        print(f"Max tokens: {self.max_tokens}")
        print('hello')
        print('world')
        print('how are you')
        self.tokenize = tokenize

        self._distribute_tokens()
        self._validate()

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

if __name__ == "__main__":
    example()
