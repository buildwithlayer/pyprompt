from .prompt_element import PromptElement
from .utils import EncodingFunc, DecodingFunc, create_render_map

__all__ = ("render_prompt",)


def render_prompt(
    prompt: PromptElement,
    encoding_func: EncodingFunc,
    decoding_func: DecodingFunc,
    props: dict | None = None,
    total_budget: int = 4096,
) -> list[dict[str, str]]:
    """
    Renders a prompt structure into OpenAI-compatible messages while managing token budget.
    """
    if props is None:
        props = dict()

    reserved_total = prompt.get_reserved(total_budget)
    if reserved_total > total_budget:
        raise ValueError("Could not reserve tokens for given budget")

    token_map = prompt.get_token_map(props, encoding_func)
    token_count = prompt.get_token_count(token_map)

    if token_count > total_budget:
        token_map = prompt.prune(total_budget, token_map, encoding_func, decoding_func)
        if token_map is None:
            raise ValueError("Could not prune prompt")

    render_map = create_render_map(token_map, decoding_func)
    return list(prompt.generate_messages(render_map))
