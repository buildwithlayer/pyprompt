from __future__ import annotations

from sys import maxsize
from typing import Generator, Callable, Any

from .utils import (
    EncodingFunc,
    DecodingFunc,
    GrowCallback,
    RenderMap,
    TokenMap,
    is_in_token_map,
    update_token_map,
    orphan_child_from_token_map,
)

__all__ = ("PromptElement",)


class PromptElement:
    """
    Core class for building hierarchical prompt structures with priority-based token management.

    Key features:
    - Nested prompt structures
    - Priority-based pruning
    - Token reservation
    - Template variable substitution

    Interacts with:
    - TextChunk: For breaking text on specific characters
    - Message types: For OpenAI message formatting
    - History: For conversation history management
    """

    def __init__(
        self,
        *children: PromptElement | str,
        priority: int = maxsize,
        pass_priority: bool = False,
        reserve: int | float | None = None,
        grow_ratio: int | float = 1,
        grow_callback: GrowCallback | None = None,
    ):
        self.children = children
        self.priority = priority
        self.pass_priority = pass_priority
        self.reserve = reserve
        self.grow_ratio = grow_ratio
        self.grow_callback = grow_callback

        self._priority_chain = None
        self._priorities = None

    def __iter__(self) -> Generator[list[int], None, None]:
        for _, descendant_indices in self.priorities:
            yield descendant_indices

    @property
    def priority_chain(self) -> list[int]:
        """
        The element's priority followed by its descendants maximum priority.
        :return: list[priority of descendants]
        """
        if self.pass_priority:
            return []

        if self._priority_chain is None:
            child_chains = []
            for child in self.children:
                if isinstance(child, PromptElement):
                    child_chains.append(child.priority_chain)
                else:
                    child_chains.append([maxsize])

            if len(child_chains) > 0:
                child_chains.sort(reverse=True)
                self._priority_chain = [self.priority] + child_chains[0]
            else:
                self._priority_chain = [self.priority]
        return self._priority_chain

    @property
    def priorities(self) -> list[tuple[list[int], list[int]]]:
        """
        The priorities of all descendants, in order of their priority chains.
        :return: list[(priority_chain, [descendant_indices])]
        """
        if self._priorities is None:
            descendants = []
            for idx, child in enumerate(self.children):
                if isinstance(child, PromptElement):
                    if child.pass_priority:
                        for priority_chain, descendant_indices in child.priorities:
                            descendants.append(
                                (priority_chain, [idx] + descendant_indices)
                            )
                    else:
                        descendants.append((child.priority_chain, [idx]))
                else:
                    descendants.append(([maxsize], [idx]))

            if not self.pass_priority:
                descendants.sort(reverse=True)

            self._priorities = descendants
        return self._priorities

    def get_token_map(self, props: dict, encoding_func: EncodingFunc) -> TokenMap:
        """
        Creates a token map from the given props.
        :param props: a dict of variables to be substituted into strings of the render map.
        :param encoding_func: A function that takes a string and returns the list of tokens.
        :return: TokenMap
        """
        token_map = dict()
        for idx, child in enumerate(self.children):
            if isinstance(child, PromptElement):
                token_map[idx] = child.get_token_map(props, encoding_func)
            elif isinstance(child, str):
                token_map[idx] = encoding_func(child.format_map(props))
            else:
                raise TypeError(f"Unsupported child type: {type(child)}")
        return token_map

    def get_reserved(self, budget: int) -> int:
        """
        Gets the total reserved token count based on the given budget.
        :param budget: the number of tokens allowed.
        :return: int
        """
        reserved = 0
        if isinstance(self.reserve, int):
            reserved += self.reserve
        elif isinstance(self.reserve, float):
            reserved += int(self.reserve * budget)

        for descendant_indices in self:
            child = self.get_child(descendant_indices)
            if not isinstance(child, PromptElement):
                continue
            reserved += child.get_reserved(budget)

        return reserved

    def get_token_count(self, token_map: TokenMap) -> int:
        """
        Get the token count for the given element.
        :param token_map: The TokenMap for this element.
        :return: int
        """
        token_count = 0
        for idx, token_value in token_map.items():
            child = self.children[idx]
            if isinstance(child, PromptElement):
                if not isinstance(token_value, dict):
                    raise TypeError(
                        f"Type mismatch: child = {type(child)}, token_value = {type(token_value)}"
                    )
                token_count += child.get_token_count(token_value)
            else:
                if not isinstance(token_value, list):
                    raise TypeError(
                        f"Type mismatch: child = {type(child)}, token_value = {type(token_value)}"
                    )
                token_count += len(token_value)
        return token_count

    def get_child(self, descendant_indices: list[int]) -> PromptElement | str:
        """
        Retrieves the child/descendant element from the given indices.
        :param descendant_indices: list[int] of indices of descendants.
        :return: PromptElement | str
        """
        if len(descendant_indices) == 1:
            return self.children[descendant_indices[0]]
        else:
            child = self.children[descendant_indices[0]]
            if not isinstance(child, PromptElement):
                raise ValueError("child must be an instance of PromptElement")
            return child.get_child(descendant_indices[1:])

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

        for descendant_indices in reversed(list(self)):
            if not is_in_token_map(descendant_indices, token_map):
                continue

            child = self.get_child(descendant_indices)
            if isinstance(child, PromptElement):
                token_map, child_token_map = orphan_child_from_token_map(
                    token_map, descendant_indices
                )
                if child.get_token_count(child_token_map) <= child.get_reserved(budget):
                    token_map = update_token_map(
                        token_map, descendant_indices, child_token_map
                    )
                    continue

                child_budget = budget - self.get_token_count(token_map)
                if child_budget <= 0:
                    value = None
                else:
                    value = child.prune(
                        child_budget, child_token_map, encoding_func, decoding_func
                    )
                token_map = update_token_map(token_map, descendant_indices, value)
            else:
                token_map = update_token_map(token_map, descendant_indices, None)

            if token_map is None:
                return None

            token_count = self.get_token_count(token_map)
            if token_count <= budget:
                return token_map

        return None

    def can_grow(self) -> bool:
        for child in self.children:
            if isinstance(child, PromptElement) and child.can_grow():
                return True
        return self.grow_callback is not None

    def get_grow_sum(self) -> float:
        total = 0.0
        for child in self.children:
            if isinstance(child, PromptElement):
                total += child.get_grow_sum()
        if self.grow_callback is not None:
            total += float(self.grow_ratio)
        return total

    def grow(
        self,
        grow_budget: int,
        token_map: TokenMap,
        encoding_func: EncodingFunc,
        decoding_func: DecodingFunc,
    ) -> tuple[TokenMap, Any | None]:
        if not self.can_grow():
            return token_map, None

        if self.grow_callback is not None:
            starting_token_count = self.get_token_count(token_map)
            goal_token_count = starting_token_count + grow_budget

            token_count = starting_token_count
            grow_state = None
            while token_count < goal_token_count:
                new_token_map, new_grow_state = self.grow_callback(
                    self, encoding_func, decoding_func, grow_state
                )
                if new_token_map is None:
                    break

                token_count = self.get_token_count(new_token_map)
                if token_count <= goal_token_count:
                    token_map = new_token_map
                    grow_state = new_grow_state

            return token_map, grow_state

        grow_sum = self.get_grow_sum()
        grow_part = grow_budget / grow_sum

        for idx, child in enumerate(self.children):
            if not isinstance(child, PromptElement) or not child.can_grow():
                continue

            child_grow_budget = int(child.get_grow_sum() * grow_part)
            new_token_map, new_grow_state = child.grow(
                child_grow_budget,
                token_map[idx],
                encoding_func,
                decoding_func,
            )
            if new_token_map is not None:
                token_map[idx] = new_token_map

        return token_map, None

    def generate_messages(self, render_map: RenderMap) -> Generator[dict, None, None]:
        """
        Generates message dicts for itself and children.

        :param render_map: The computed render map.
        :return: message dictionaries.
        """
        for idx, child in enumerate(self.children):
            if idx in render_map:
                yield from child.generate_messages(render_map[idx])
