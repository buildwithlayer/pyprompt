from .prompt_element import PromptElement
from .utils import DecodingFunc, EncodingFunc, create_render_map

__all__ = ("render_prompt",)


def render_prompt(
    prompt: PromptElement,
    encoding_func: EncodingFunc,
    decoding_func: DecodingFunc,
    total_budget: int = 4096,
) -> list[dict[str, str]]:
    """
    Renders a prompt structure into OpenAI-compatible messages while managing token budget.
    """

    reserved_total = prompt.get_reserved(total_budget)
    if reserved_total > total_budget:
        raise ValueError("Could not reserve tokens for given budget")

    token_map = prompt.get_token_map(encoding_func)
    token_count = prompt.get_token_count(token_map)

    if token_count > total_budget:
        token_map = prompt.prune(total_budget, token_map, encoding_func, decoding_func)
        if token_map is None:
            raise ValueError("Could not prune prompt")

    token_count = prompt.get_token_count(token_map)
    grow_budget = total_budget - token_count
    if grow_budget > 0:
        token_map, _ = prompt.grow(
            grow_budget,
            token_map,
            encoding_func,
            decoding_func,
        )

    render_map = create_render_map(token_map, decoding_func)
    return list(prompt.generate_messages(render_map))
