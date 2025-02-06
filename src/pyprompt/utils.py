from typing import Callable, Union, Generator

__all__ = (
    "RenderMap",
    "TokenMap",
    "BudgetMap",
    "EncodingFunc",
    "DecodingFunc",
    "orphan_child_from_token_map",
    "update_token_map",
    "create_render_map",
    "is_in_token_map",
    "iter_render_map",
)

RenderMap = dict[int, Union[str, 'RenderMap']]
TokenMap = dict[int, Union[list[int], 'TokenMap']]
BudgetMap = dict[int, Union[int | float, 'BudgetMap']]
EncodingFunc = Callable[[str], list[int]]
DecodingFunc = Callable[[list[int]], str]


def orphan_child_from_token_map(token_map: TokenMap, descendant_indices: list[int]) -> tuple[TokenMap, TokenMap | list[int]]:
    if len(descendant_indices) == 1:
        parent_token_map = token_map
        child_token_map = token_map.pop(descendant_indices[0])
        return parent_token_map, child_token_map

    sub_token_map = token_map.pop(descendant_indices[0])
    sub_parent_token_map, sub_child_token_map = orphan_child_from_token_map(sub_token_map, descendant_indices[1:])
    if len(sub_parent_token_map) > 0:
        token_map[descendant_indices[0]] = sub_parent_token_map
    return token_map, sub_child_token_map


def update_token_map(token_map: TokenMap, descendant_indices: list[int], value: TokenMap | list[int] | None) -> TokenMap:
    if value is None:
        return prune_from_token_map(token_map, descendant_indices)

    if len(descendant_indices) == 1:
        token_map[descendant_indices[0]] = value
        return token_map

    if descendant_indices[0] not in token_map:
        token_map[descendant_indices[0]] = dict()
    token_map[descendant_indices[0]] = update_token_map(token_map[descendant_indices[0]], descendant_indices[1:], value)
    return token_map


def prune_from_token_map(token_map: TokenMap, descendant_indices: list[int]) -> TokenMap | None:
    if len(descendant_indices) == 1:
        token_map.pop(descendant_indices[0], None)
        if len(token_map) == 0:
            return None
        return token_map

    child_token_map = prune_from_token_map(token_map[descendant_indices[0]], descendant_indices[1:])
    if child_token_map is None:
        token_map.pop(descendant_indices[0], None)
        if len(token_map) == 0:
            return None
        return token_map
    token_map[descendant_indices[0]] = child_token_map
    return token_map


def is_in_token_map(descendants: list[int], token_map: TokenMap) -> bool:
    sub_token_map = token_map
    for idx in descendants:
        if idx not in sub_token_map:
            return False
        sub_token_map = sub_token_map[idx]
    return True


def create_render_map(token_map: TokenMap, decoder: Callable[[list[int]], str]) -> RenderMap:
    render_map = dict()
    for key, value in token_map.items():
        if isinstance(value, list):
            for v in value:
                if not isinstance(v, int):
                    raise TypeError(f"Unsupported type in value: {type(v)}")
            render_map[key] = decoder(value)
        elif isinstance(value, dict):
            render_map[key] = create_render_map(value, decoder)
        else:
            raise TypeError(f"Unsupported value type: {type(value)}")
    return render_map


def iter_render_map(render_map: RenderMap) -> Generator[str, None, None]:
    for _, value in render_map.items():
        if isinstance(value, str):
            yield value
        else:
            yield from iter_render_map(value)
