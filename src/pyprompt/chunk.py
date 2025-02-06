from .prompt_element import PromptElement
from .utils import EncodingFunc, DecodingFunc, TokenMap

__all__ = ("Chunk",)


class Chunk(PromptElement):
    def prune(
            self,
            token_map: TokenMap,
            descendant_indices: list[int],
            encoding_func: EncodingFunc,
            decoding_func: DecodingFunc,
    ) -> TokenMap | None:
        return None
