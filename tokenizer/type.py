from enum import Enum
from typing import Union

class DefaultSpecialToken(Enum):
    UNKNOWN_TOKEN = 0,
    PADDING_TOKEN = 1,
    WHITESPACE_TOKEN = 2

SpecialToken = str
SpecialTokenSet = list[Union[DefaultSpecialToken, SpecialToken]]

Jamo = list[
    str,
    str,
    str
]

JamoList = list[Jamo]