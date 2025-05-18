from enum import Enum
from typing import Any

import numpy as np
from numpy._typing import _64Bit


class DefaultSpecialToken(Enum):
    _PAD = 0,
    _UNK = 1


# [ㄱ, ㅏ, ㅁ]
JamoChar = list[3, str]


# [ [ㄱ, ㅏ, ㅁ] , [ㅈ,ㅏ,PAD] ]
JamoCharArray = list[JamoChar]


# [
#   [ㄱ, ㅏ, ㅁ] , [ㅈ,ㅏ,PAD],
#   [ㄱ, ㅗ, PAD] , [ㄱ,ㅜ,PAD] , [ㅁ,ㅏ,PAD],
#   ...
# ]
JamoCharArrayDataset = list[JamoCharArray]


# [one-hot(ㄱ) , one-hot(ㅏ) , one-hot(ㅁ)]
JamoVector = np.ndarray[(Any, 3, Any), np.dtype[np.floating[_64Bit]]]


# [ [one-hot(ㄱ) , one-hot(ㅏ) , one-hot(ㅁ)] , [one-hot(ㅈ) , one-hot(ㅏ) , zero() ] ]
JamoMatrix = np.ndarray[(Any, JamoVector), np.dtype[np.floating[_64Bit]]]


# [
#   [one-hot(ㄱ) , one-hot(ㅏ) , one-hot(ㅁ)] , [one-hot(ㅈ) , one-hot(ㅏ) , zero() ],
#   [one-hot(ㄱ) , one-hot(ㅗ) , zero()] , [one-hot(ㄱ) , one-hot(ㅜ) , zero()] , [one-hot(ㅁ) , one-hot(ㅏ) , zero()]
#   ...
# ]
JamoMatrixDataset = np.ndarray[(Any,JamoMatrix), np.dtype[np.floating[_64Bit]]]
