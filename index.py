# import torch.cuda
from transformers import PreTrainedTokenizerFast

from corpus.data_loader import DataLoader
from corpus.data_normalizers import DataNormalizers
from tokenizer.jamo.tokenizer import JamoTokenizer
from tokenizer.jamo.vocab import Vocab
from tokenizer.pre_training import pre_training_tokenizer, load_data

###############################################
# process const variable
###############################################
from dotenv import load_dotenv
load_dotenv()

contents = [
    "개웃기네ㅋㅋㅋㅋㅋㅋㅋ",
    "와아아아아앙 우오오오오아앙",
    "이거 개쩜 https://arxiv.org/pdf/2004.05150 진짜 미쳤다 https://arxiv.org/pdf/2004.05150",
    "실화냐........",
    "와우와우와우와우",
    "어쩔어쩔어쩔어쩔",
    "응아니야응아니야응아니야",
    "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
    "웃겨ㅕㅕㅕㅕㅕㅕㅕㅕㅕㅕ",
    "빼에에에에에에에에에에엑",
    "와아아아아아아아아아아아",
    "응ㅋ어쩔어쩔어쩔어쩔어쩔",
    "어쩔티비어쩔티비어쩔티비",
    "ㅅㄱㄹㅅㄱㄹㅅㄱㄹ"
]

def test_tokenizer():
    # result = load_data("read")
    # tokenizer = pre_training_tokenizer(result)

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="./tokenizer/data/tokenizer.json",
        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    while True:
        data = input("\n입력:")
        print(wrapped_tokenizer.encode(data))
        print(wrapped_tokenizer.tokenize(data))


def test_Corpus_normalizers():

    dataloader = DataLoader()
    # contents = dataloader.load_csv(path="./corpus/test_data/type1.csv", is_header=False)

    corpus = DataNormalizers()
    corpus.add_all_method()

    # dataframe > bag 타입 변환 시 tuple 단일 원소 튜플이 되므로 [0]번째 원소 출력하는 로직을 추가해야함
    # y =contents.map(lambda x: corpus.filtering_normalize(x[0])).compute()

    corpus.set_corpus(contents)
    y = corpus.compute_normalize()
    print(y)


def test_tokenizer_train():
    JamoTokenizer().train_tokenzier(contents)

if __name__ == '__main__':
    # print("cuda status: ", torch.cuda.is_available())

    Vocab("tokenizer/jamo/load_test_vocab.json").load_vocab()
    # test_Corpus_normalizers()

    test_tokenizer_train()

