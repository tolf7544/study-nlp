from tokenizer.builder import TokenizerBuilder
from tokenizer.jamo_tokenizer import JamoTokenizer
from util.type import NormalizerMethod

sentence_corpus = ["안녕하세요!!!"]

def test_1():
    vocab_builder = TokenizerBuilder()

    tokenizer = vocab_builder.set_special_token().set_corpus(sentence_corpus).set_normalizing().build_tokenizer()

    x = tokenizer.tokenize("안녕하세요###@$!&$%#", length=10, padding=True, truncation=False)
    print(tokenizer.vocab.get_vocab())
    x = tokenizer.encode(x)["encode"]
    print(x)
    x = tokenizer.decode(x)
    print(x)
    x = tokenizer.combine(x)
    print(x)

def test_2():
    vocab_builder = TokenizerBuilder()
    tokenizer = vocab_builder.set_special_token().set_corpus(sentence_corpus).set_normalizing().multi_processing()
if __name__ == '__main__':
    # test_1() # 통과
    test_2()
    pass