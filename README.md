# PyPrompt
## LLM Prompt Validation and Manipulation

This project provides functionality for creating prompts for LLMs, including their validation and manipulation. This includes functionality for limiting the number of tokens used for LLM prompts and distributing these tokens among the parts of the LLM message.

Note that the tokenizer used here may not be exactly the same as the one used by the LLM, so the actual number of tokens used may be greater than or less than this module estimates. If using more tokens the the context window is problematic, you can add a margin to this module's estimate by simply passing in a smaller number of tokens than the context window.

## Dependencies

The required packages can be installed by running `poetry install`. This depends on `poetry` being installed locally; instructions can be found [here](https://python-poetry.org/docs/).

## Usage

For direction on how to use this module, please see function `example` in `pyprompt/prompt_validation.py` and read the appropriate docstrings as necessary.

## Tests

The test suite can be run using `poetry run pytest`.