# PyPrompt

## Installation

`pip install pyprompt`

## Usage

```python
from pyprompt import *

prompt = PromptElement()
```

### Properties

For pruning (removing text when you have more tokens than budgeted), the `priority`, `pass_priority`, and `reserve`
fields are used.

- `priority`: This determines the order that elements are pruned. The elements are pruned from the lowest priority to
  highest, and the default value is max priority.
- `pass_priority`: Normally, priority is compared amongst sibling elements. If `pass_priority` is `True`, then that
  element's children are compared to its siblings. The default value is `False`.
- `reserve`: This keyword, when set, reserves a specified number of tokens from the total budget of the prompt. It can
  be set as an `int` (e.g. 60, which would reserve 60 tokens) or a `float` (e.g. 1/3, which would reserve one-third of
  the total token budget). When pruning, this element can be pruned down to its reserve amount, but not past it. The
  default value is `None`.

For growing (adding more text when the budget allows it), the `grow_ratio` and `grow_callback` are used. For an element
to grow into the unused budget, it must have a `grow_callback`.

- `grow_ratio`: This determines what portion of the unused budget is given to this element. The portion of `grow_ratio`
  to the sum of all `grow_ratios` dictates how many tokens from the unused budget are allocated to this element. For
  example, if three elements have `grow_ratios` of 1, 2, and 3 and the unused token budget is 60 tokens, then they will
  get 10, 20, and 30 tokens to grow, respectively. The `grow_ratio` can be an `int` or `float`, and the default value is
  `1`.
- `grow_callback`: This is a function that creates a new `TokenMap` for a `PromptElement` that uses up more of the
  budget, if possible. The function's signature is as follows:
    - Inputs:
        - `PromptElement`: The element to grow.
        - `encoding_func`: The function to encode strings to tokens.
        - `decoding_func`: The function to decode tokens to strings.
        - `Any | None`: The state of the function (`None` the first time the function is called, then the output of the
          previous call).
    - Outputs:
        - `TokenMap | None`: The `TokenMap` of the `PromptElement` after growing, or `None` if it cannot grow anymore.
        - `Any | None`: The state of the function, used in the next grow call.
- It is ill-advised to have elements with `grow_callback` as descendants of another element with `grow_callback`.
  `grow_callback` should only be set on leaf nodes.