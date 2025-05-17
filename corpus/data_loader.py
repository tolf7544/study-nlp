import typing
import dask.dataframe
import dask.dataframe as dd
from dask.bag import Bag
from corpus.type import Path


class DataLoader:
    block_size: int

    #2**31 = 2gb
    #2**30 = 1gb
    #2**29 = 512mb
    #2**28 = 256mb


    def __init__(self, data_block_size: typing.Optional[int] = 2**28):
        r"""_block_size= 2**28(256mb) default | 2**29(512mb) | 2**30(1gb) ... """
        self.block_size = data_block_size



    # https://stackoverflow.com/questions/58541722/what-is-the-correct-way-in-python-to-annotate-a-path-with-type-hints
    def load_csv(self, path: Path, is_header:bool =True)-> Bag:
        header = "infer"

        if not is_header:
            header = None
        temp = dask.dataframe.DataFrame(dd.read_csv(path, self.block_size, header=header))
        return  temp.to_bag()

    def load_json(self, path: Path)-> Bag:
        temp = dask.dataframe.DataFrame(dd.read_json(path, self.block_size))
        return  temp.to_bag()

    def load_txt(self, path: Path, is_header:bool =True)-> Bag:
        header = "infer"

        if not is_header:
            header = None
        temp = dask.dataframe.DataFrame(dd.read_json(path, self.block_size, header=header))
        return temp.to_bag()