from .prompt_element import PromptElement
from .utils import DecodingFunc, EncodingFunc, TokenMap, update_token_map

__all__ = ("TextChunk",)


class TextChunk(PromptElement):
    """
    Handles breakable text content with token-aware pruning capabilities.

    Key features:
    - Character-based text splitting
    - Token-aware content pruning
    - Maintains semantic boundaries when possible
    """

    def __init__(
        self,
        *children: str,
        break_on: str | None = None,
        **kwargs,
    ):
        super().__init__(*children, **kwargs)
        self.break_on = break_on

    def prune(
        self,
        budget: int,
        token_map: TokenMap,
        encoding_func: EncodingFunc,
        decoding_func: DecodingFunc,
    ) -> TokenMap | None:
        token_count = self.get_token_count(token_map)
        if token_count <= budget:
            return token_map

        for idx, child in enumerate(self.children):
            tokens = token_map[idx]
            if self.break_on is None:
                while len(tokens) > 1:
                    tokens = tokens[:-1]
                    token_map[idx] = tokens

                    token_count = self.get_token_count(token_map)
                    if token_count <= budget:
                        return token_map
            else:
                text = decoding_func(tokens)
                parts = text.split(self.break_on)
                while len(parts) > 1:
                    parts = parts[:-1]
                    text = self.break_on.join(parts)
                    tokens = encoding_func(text)
                    token_map[idx] = tokens

                    token_count = self.get_token_count(token_map)
                    if token_count <= budget:
                        return token_map

            token_map = update_token_map(token_map, [idx], None)
            if token_count <= budget:
                return token_map

        return None
