import os
from typing import Union
from enum import Enum, auto
import numpy as np


# enum class - _정리
class NormalizerOption(Enum):
    CLEAN_TEXT = auto()
    # REMOVE_URL = auto()
    # REMOVE_EMAIL = auto()
    REMOVE_REPETITION_CHAR = auto()


class JamoEncodingType(Enum):
    TOKEN_VECTOR = auto()
    r"""일반적인 token로 구성된 일차원 백터"""
    JAMO_VECTOR = auto()
    r"""sparse vector로 (3, jamo one hot encoding vector)형태인 2차원 형태 백터"""

Corpus = Union[list, np.ndarray]
Path = Union[str, os.PathLike]
