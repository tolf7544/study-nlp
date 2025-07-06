from enum import Enum
from typing import Type, Optional

from jinja2.nodes import Literal

JamoSet:Type = list[str,str,str]

class NormalizerMethod(Enum):
        REMOVE_CONTROL_CHAR = 1,
        REDUCE_WHITESPACE = 2,
        REMOVE_ACCENT = 3,
        REMOVE_REPETITION_CHAR = 4,
        NFK = 5,