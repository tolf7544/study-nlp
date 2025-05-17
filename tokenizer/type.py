from enum import Enum

import numpy as np


class DefaultSpecialToken(Enum):
    PAD = 0,
    UNK = 1

class SpecialToken(str):...
class CustomSpecialToken(str):...

class JamoChar(np.ndarray[
    str,
    str,
    str
]): ...

class JamoCharArray(np.ndarray[JamoChar]):...
class JamoCharArrayDataset(np.ndarray[JamoCharArray]):...

class JamoVector(np.ndarray[
    np.ndarray,
    np.ndarray,
    np.ndarray
]):...
class JamoMatrix(np.ndarray[JamoVector]):...
class JamoMatrixDataset(np.ndarray[JamoMatrix]):...