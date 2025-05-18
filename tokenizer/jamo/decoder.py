from tkinter.messagebox import RETRY

import numpy as np
from jamo import is_jamo

from tokenizer.jamo.vocab import Vocab
from tokenizer.type import DefaultSpecialToken, JamoVector, JamoChar, JamoCharArray, JamoMatrix


class JamoTokenizerDecoder:
    vocab: Vocab
    sentence_length: int
    is_padding: bool
    is_truncation: bool

    def __init__(self,
                 vocab: Vocab,
                 sentence_length: int = 200,
                 is_padding: bool = False,
                 is_truncation: bool = False
                 ):
        self.vocab = vocab
        self.sentence_length = sentence_length
        self.is_padding = is_padding
        self.is_truncation = is_truncation

    def __decode_one_hot_vector(self, jamo_vector: JamoVector ) -> JamoVector:
        one_dim_vector = np.zeros((3,), dtype=np.object_)

        for i in range(3):
            index = np.where(jamo_vector[i] > 0)
            key = 0

            if index[0].__len__() != 0:
                key = int(index[0])

            one_dim_vector[i] = self.vocab.key_2_token(key)
        return one_dim_vector


    def decode_one_hot_vector_matrix(self, jamo_matrix: JamoMatrix) -> JamoMatrix:
        one_dim_vector = np.zeros((jamo_matrix.__len__(),3), dtype=np.object_)
        for i in range(jamo_matrix.__len__()):
            one_dim_vector[i] = self.__decode_one_hot_vector(jamo_matrix[i])
        return one_dim_vector