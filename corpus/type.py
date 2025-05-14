import os
from typing import Union, Optional
from enum import Enum, auto
import numpy
import torch

# enum class - _정리
class NormalizationMethod(Enum):
    CLEAN_TEXT = auto()
    # REMOVE_URL = auto()
    # REMOVE_EMAIL = auto()
    REMOVE_REPETITION_CHAR = auto()


class JamoEncodingType(Enum):
    TOKEN_VECTOR = auto()
    r"""일반적인 token로 구성된 일차원 백터"""
    JAMO_VECTOR = auto()
    r"""sparse vector로 (3, jamo one hot encoding vector)형태인 2차원 형태 백터"""

TypeCorpus = Union[list[str],Optional[Union[numpy.ndarray]]]
TypePath = Union[str, os.PathLike]
