import numpy as np

from tokenizer.builder import TokenizerBuilder
from tokenizer.jamo_tokenizer import JamoTokenizer
from tokenizer.vocab import Vocab
from util.type import JamoSet

test_dict = {"ㄱ":0, "ㄴ": 1, "ㄷ": 2, "<pad>": 3, "<unk>": 4, "<mask>": 5, "ㅇ": 6, "ㅏ": 7, "ㄹ": 8, "ㅁ": 9}
encoded_contents = [1, 2, 3, 5, 4, 4, 4]
sentence = "가나다라마12"

# vocab = Vocab(test_dict,padding_token="<pad>", unknown_token="<unk>")
tokenizer_builder = TokenizerBuilder()
tokenizer = tokenizer_builder.set_special_token(["<mask>"]).set_corpus(sentence).set_normalizing().build_tokenizer() # builder test


repetition_word_include_sentence = "aaaaaaaa ababab abcabcabc a1a1a1 bzsdcabzsdcabzsdcabzsdca !!!!!"

def test_1(): # tokenize - detail test
    print("normal test - len: 10, pad: F, tru: F")
    a = tokenizer.tokenize(sentence=sentence, length=10, padding=False, truncation=False )
    print(a)
    print("padding test - len: 10, pad: T, tru: F/T")
    a = tokenizer.tokenize(sentence=sentence, length=10, padding=True, truncation=False )
    b = tokenizer.tokenize(sentence=sentence, length=10, padding=True, truncation=True )
    print(a)
    print("truncation 영향: ", b != a)  # False

    print("truncation test - len: 10, pad: F/T, tru: T")
    a = tokenizer.tokenize(sentence=sentence, length=10, padding=True, truncation=True )
    b = tokenizer.tokenize(sentence=sentence, length=10, padding=False, truncation=True )
    print(a)
    print("padding 영향: ", b != a) # True

def test_2(): # normalize / tokenize
    print("normal test - len: 10, pad: F, tru: F")

    s = tokenizer.normalize(repetition_word_include_sentence)
    print("정규화 결과 값: ", s)
    s = tokenizer.tokenize(sentence=s, length=50, padding=False, truncation=False )
    print("형태소 분해: ", s)

def test_2_5(s):

    print("normal test - len: 10, pad: F, tru: F")
    s = tokenizer.tokenize(sentence=s, length=10, padding=True, truncation=False )
    print("형태소 분해: ", s)
    return s

def test_3():
    a = test_2_5(sentence)

    a = tokenizer.combine(a)
    print("형태소 결합: ", a)
    print("형태소 결합 - 일치 여부: ", a == sentence)

def test_4():
    a = test_2_5(sentence)
    print("normal test - encode")
    print(tokenizer.encode(token_list=a, return_attention_mask=True))
    a1 = tokenizer.encode(token_list=a)["encode"]
    print("encode result: ",end="")
    print(np.array(a1))

    a1 = tokenizer.decode(a1)
    print("decode result: ", a1)
    print(" is same before encode, after decode: ", a == a1)

    a1 = tokenizer.combine(a1)
    print("combine result:", a1)
    print("is same: ", sentence == a1)

def test_5():
    tokenizer.save_vocab("./vocab.json")

def test_6():
    print("normal test - transform default token")
    test_dict["[PAD]"] = test_dict.pop("<pad>")
    test_dict["[UNK]"] = test_dict.pop("<unk>")
    print(test_dict)
    # vocab = Vocab(test_dict,["<mask>"], padding_token="[PAD]", unknown_token="[UNK]")
    _tokenizer_builder = TokenizerBuilder()
    _tokenizer = _tokenizer_builder.set_special_token(special_token=["<mask>"],padding_token="[PAD]", unknown_token="[UNK]").set_corpus(
        sentence).set_normalizing().build_tokenizer()  # builder test

    print("normal test - save vocab")
    _tokenizer.save_vocab("./vocab.json")

    print("normal test - len: 10, pad: F, tru: F")
    s = _tokenizer.tokenize(sentence=sentence, length=10, padding=False, truncation=False )
    print("형태소 분해: ", s)

def test_7():
    __tokenizer = JamoTokenizer(vocab_path="../test/vocab.json")
    s = tokenizer.tokenize(sentence=sentence, length=10, padding=False, truncation=False )
    print(s)
if __name__ == '__main__':
    # test_1() # 성공 # 빌더 성공
    # test_2() # 성공 # 빌더 성공
    # test_3() # 성공 # 빌더 성공
    test_4() # 성공 # 빌더 성공
    # test_5() # 성공 # 빌더 성공
    # test_6() # 성공 # 빌더 성공
    # test_7() # 성공
    pass