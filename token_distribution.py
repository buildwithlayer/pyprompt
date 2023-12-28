import tiktoken
from math import floor

class Block:
    '''
    Represents a section of the prompt, including its name and its contents.
    To create a block with more specific behavior, derive from this class.
    '''
    def __init__(self, name, data, to_text):
        '''
        Initializes this block object.
        
        Args:
            name: The name of the block.
            data: The contents of the block.
            to_text: The text representation of the data. If the data is 
            conceptually empty, an empty string should be returned.
        '''
        self.name = name
        self.data = data
        self.to_text = to_text
    
    def reduce(self, factor):
        '''
        Creates a new block with content reduced by the given factor.
        '''
        min_idx = len(self.data) - floor(len(self.data) * factor)
        return Block(self.name, self.data[min_idx:], self.to_text)

    def __str__(self):
        return self.to_text(self.data)
       
def order_blocks(order_by, to_order):
    '''
    Orders the blocks in the set to_order according to the 
    order of the blocks in list order_by.
    '''
    block_index = {block.name: i for i, block in enumerate(order_by)}
    return sorted(to_order, key=lambda block: block_index[block.name])

def reduce_single_block(tokenize, block, max_tokens):
    '''
    Creates a block that uses no more than the given number of 
    tokens to represent its contents with the given tokenize function.
    '''
    num_tokens = len(tokenize(str(block)))
    while num_tokens > max_tokens:
        block = block.reduce(max_tokens / num_tokens)
        num_tokens = len(tokenize(str(block)))
    return max_tokens - num_tokens, block

def reduce_blocks_simple(tokenize, blocks, max_tokens):
    '''
    Creates a set of blocks that use no more than the given number of
    tokens to represent their contents with the given tokenize function.
    '''
    result_blocks = set()
    # sort blocks by name to ensure deterministic ordering
    for block in sorted(blocks, key=lambda block: block.name):
        max_tokens, new_block = reduce_single_block(tokenize, block, max_tokens)
        result_blocks.add(new_block)
    return max_tokens, result_blocks

class ArbitrationFunction:
    '''
    Creates a callable object that arbitrates among blocks with names in a given set.
    To use, pass in tokenize function, a set of blocks, and the max number of tokens
    that may be distributed amongst the blocks. The function returns a tuple of the
    number of tokens remaining and the set of blocks matching the internal set of names, 
    modified to use no more than the maximum number of tokens.
    '''
    def __init__(self, block_names, function):
        '''
        Creates an arbitration function that arbitrates among blocks with the given names.

        Args:
            block_names: The set of block names that this arbitration function arbitrates.
            function: The arbitration function.
        '''
        self.block_names = block_names
        self.function = function

    def __call__(self, tokenize, blocks, max_tokens):
        '''
        Calls the arbitration function on the given blocks.
        '''
        return self.function(
            tokenize,
            {block for block in blocks if block.name in self.block_names},
            max_tokens
        )

class Arbitrator:
    def __init__(self, arbitration_functions, default_function):
        '''
        Creates an arbitrator object for distributing tokens among blocks.

        Args:
            arbitration_functions: A list of ArbitrationFunctions which will
            be applied to the blocks in order.
            defaut_function: The function to use for blocks not handled by
            any of the arbitration functions.
        '''
        self.arbitration_functions = arbitration_functions
        self.default_function = default_function       
        self.arbitrated_blocks = set().union(*(arbitrator_function.block_names for arbitrator_function in arbitration_functions))

    def distribute_tokens(self, tokenize, blocks, max_tokens):
        '''
        Distributes tokens amongst the given blocks. Any blocks not handled by an
        arbitration function are handled by the default function.

        Args:
            tokenize: The function used to tokenize the blocks.
            blocks: The list of blocks among which to distribute tokens.
            max_tokens: The maximum number of tokens to use.
        '''
        tokens_remaining = max_tokens
        result_blocks = set()

        for arbitration_function in self.arbitration_functions:
            tokens_remaining, arbitrated_blocks = arbitration_function(
                tokenize, blocks, tokens_remaining)
            result_blocks.update(arbitrated_blocks)

        tokens_remaining, arbitrated_blocks = self.default_function(
            tokenize,
            {block for block in blocks if block.name not in self.arbitrated_blocks},
            tokens_remaining,
        )
        result_blocks.update(arbitrated_blocks)

        return tokens_remaining, order_blocks(order_by=blocks, to_order=result_blocks)
    
def example():
    '''
    An example demonstrating how to use this module.
    '''
    encoding_name = "cl100k_base"
    tokenize = tiktoken.get_encoding(encoding_name).encode
    blocks = [
        Block(name='system_prompt', data='aaa', to_text=str),
        Block(name='user_input', data='bbb', to_text=str),
        Block(name='chat_history', data='ccc', to_text=str),
        Block(name='actions', data='ddd', to_text=str),
    ]
    arbitrator = Arbitrator(
        arbitration_functions=[
            ArbitrationFunction({'system_prompt'}, reduce_blocks_simple),
            ArbitrationFunction({'user_input'}, reduce_blocks_simple),
            ArbitrationFunction({'chat_history'}, reduce_blocks_simple),
            ArbitrationFunction({'actions'}, reduce_blocks_simple),
        ],
        default_function=reduce_blocks_simple
    )
    tokens_remaining, result_blocks = arbitrator.distribute_tokens(
        tokenize=tokenize,
        blocks=blocks,
        max_tokens=10,
    )

    print(f'There are {tokens_remaining} tokens remaining.')

    for block in result_blocks:
        print(str(block))

if __name__ == "__main__":
    example()