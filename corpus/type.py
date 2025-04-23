import os
from typing import Union, Optional
from enum import Enum, auto
import numpy
import torch

# enum class - _정리
class NormalizationMethod(Enum):
    REMOVE_LINEBREAKS = auto()
    REMOVE_WHITESPACE = auto()
    REMOVE_URL = auto()
    REMOVE_REPETITION_CHAR = auto()


TypeCorpus = Union[list[str],Optional[ Union[numpy.ndarray, torch.Tensor]]]
TypePath = Union[str, os.PathLike]