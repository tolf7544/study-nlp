from enum import Enum, auto
from typing import Optional, Union

from dask.dataframe import DataFrame


class ErrorType(Enum):
    OVERFLOW = auto()
    DATA_ISSUE = auto()
    MISS_MATCHING = auto()

# 클래스 상태 정의 타입
class CorpusLoaderStatus(Enum):
    SUCCESS = auto()
    ERROR_CORPUS_IMPORTING = auto()
    ERROR_CORPUS_FORMAT = auto()

class CorpusNormalizerStatus(Enum):
    SUCCESS = auto()
    FAILED_REPETITION_FILTERING = auto()
    FAILED_WHITESPACE_FILTERING = auto()
    SUCCESS_BUT_ISSUE_OCCURRED = auto()


# 리턴 결과 타입
class CorpusLoaderStatusResult():
    def __init__(self, _status: CorpusLoaderStatus,
                 _contents: DataFrame,
                 _issue: Optional[ErrorType] = None ):
        self.status = _status,
        self.issue = _issue,
        self.contents = _contents

    def __repr__(self):
        return f"status: {self.status[0]},\nissue: {self.issue[0]}\ncontents: {self.contents}"

class NormalizerMethodResult():
    def __init__(self, _status: CorpusNormalizerStatus,
                 _contents: Optional[Union[str,list[str]]],
                 _issue: Optional[ErrorType] = None ):
        self.status = _status,
        self.issue = _issue,
        self.contents = _contents

    def __repr__(self):
        return f"status: {self.status[0]},\nissue: {self.issue[0]}\ncontents: {self.contents}"