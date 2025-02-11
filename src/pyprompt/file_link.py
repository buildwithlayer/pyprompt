from pathlib import Path

from .text_chunk import TextChunk
from .utils import EncodingFunc, TokenMap

__all__ = ("FileLink",)


class FileLink(TextChunk):
    def __init__(
        self,
        *children: str | Path,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _children = []
        for idx, child in children:
            if isinstance(child, str):
                _children.append(Path(child))
            elif isinstance(child, Path):
                _children.append(child)
            else:
                raise TypeError(f"Children must be a Path or str, not {type(child)}")
        self.children = tuple(_children)

    def get_token_map(self, props: dict, encoding_func: EncodingFunc) -> TokenMap:
        token_map = dict()
        for idx, child in enumerate(self.children):
            with open(child, "r") as f:
                token_map[idx] = encoding_func(f.read())
        return token_map
