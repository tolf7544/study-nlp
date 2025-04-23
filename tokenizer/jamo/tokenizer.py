class JamoTokenizer():
    sentence_length: int
    padding: bool
    truncation: bool

    def __init__(self, _sentence_length: int = 200, _padding: bool = True, _truncation: bool = True):
        self.padding = _padding
        self.sentence_length = _sentence_length
        self.truncation = _truncation

    def __zero_sparse_vector(self):
        ...