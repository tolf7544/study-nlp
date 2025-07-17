import dask.bag
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from tokenizer.builder import TokenizerBuilder
from tokenizer.loader import DataReader


def vocab_train(dataset):

    TokenizerBuilder().set_special_token().set_corpus(dataset).set_normalizing().build_tokenizer()

if __name__ == '__main__':
    bag = DataReader().load_txt("corpus/*.txt", is_header=True)
    vocab_train(bag.map(lambda x: x[1]).compute())