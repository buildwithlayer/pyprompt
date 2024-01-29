from typing import Dict, List, Union

__all__ = ("JSON_NUMBER", "JSON_ARRAY", "JSON_OBJECT", "JSON_TYPE")

JSON_NUMBER = Union[int, float]
JSON_ARRAY = List["JSON_TYPE"]
JSON_OBJECT = Dict[str, "JSON_TYPE"]
JSON_TYPE = Union[JSON_NUMBER, str, bool, None, JSON_ARRAY, JSON_OBJECT]
