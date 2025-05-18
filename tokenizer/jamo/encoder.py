import numpy as np
from jamo import is_jamo

from tokenizer.jamo.vocab import Vocab
from tokenizer.type import DefaultSpecialToken, JamoVector, JamoChar, JamoCharArray, JamoMatrix


class JamoTokenizerEncoder:
    vocab: Vocab

    def __init__(self,
                 vocab: Vocab
                 ):
        self.vocab = vocab

    def zero_one_hot_vector(self):
        return np.zeros((3, self.vocab.length), dtype=np.float64)

    def __encode_one_hot_vector(self, jamo_char: JamoChar) -> JamoVector: # tokenizer를 거치지 않은 입력값은 오류를 발생 시킬 수 있음( is_jamo()에서 사용되는 ord() 함수에서 오류 발생 )
        zero_vector = np.zeros((3,self.vocab.length), dtype=np.float64)
        for i in range(3):
            token_index = self.vocab.token_2_key(jamo_char[i])
            if token_index > self.vocab.special_token_ids[-1] and not is_jamo(jamo_char[0]):
                #sort된 vocab에서는 special token은 0~N 위치까지 차지한다. 이때 가장 마지막에 존재하는 speical token보다 index가 클 시 special을 제외하고 검색을 수행한다.
                for i in range(3):
                    zero_vector[i][token_index] = np.float64(1/3)
                break # early breaking
            elif token_index == DefaultSpecialToken._PAD:
                continue
            else:
                zero_vector[i][token_index] = np.float64(1)
        return zero_vector

    def encode_one_hot_vector_matrix(self,
                                     jamo_char_array: JamoCharArray,
                                     sentence_length: int,
                                     padding: bool,
                                     truncation: bool) -> JamoMatrix:
        zero_vector_matrix = np.zeros(( sentence_length, 3, self.vocab.length), dtype=np.float64)

        # a: 제한 길이가 입력 값 길이 보다 길때
        # b: padding이 존재할 때
        # c: truncation이 존재 할때
        # a + b
        # a' + c
        for i in range(sentence_length):
            if i > jamo_char_array.__len__()-1:
                zero_vector_matrix[i] = self.zero_one_hot_vector()
            else:
                zero_vector_matrix[i] = self.__encode_one_hot_vector(jamo_char_array[i])

        return zero_vector_matrix
