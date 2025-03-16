import csv
import json
import os
import re

import dask.bag

from corpus.type import CorpusLoaderStatusResult, CorpusLoaderStatus, ErrorType
import dask.dataframe as dd

class CorpusLoader():
    block_size = 2 ** 29  # 512mb

    def __init__(self):
        with open("corpus/info.json",encoding="utf-8", mode="r") as file:
            info = json.load(file)
            self.corpus_child_folders = ["labeled","unlabeled","normalized_data"]

            if info["folders"] != self.corpus_child_folders:
                exit("corpus/info.json keys are miss-matching. need checking info.json writing format\nexit location: corpus/data_loader.py")

            self.__base_path = info["base_path"]
            self.__corpus = info["corpus"]
            self.__corpus_keys = info["corpus"]["keys"]


    def open_file(self, path, file_name) -> CorpusLoaderStatusResult:
        if re.search('json', file_name) != None:
            json_data = dd.read_json(path, blocksize=self.block_size) #=2**30는 1gb / 2**28는 약 256mb
            return CorpusLoaderStatusResult(
                _status=CorpusLoaderStatus.SUCCESS,
                _contents=json_data)
        elif re.search('csv', file_name) != None:
            csv_data = dd.read_csv(path, blocksize=self.block_size, header=None) # heaeder는 비워져 있음

            return CorpusLoaderStatusResult(
                _status=CorpusLoaderStatus.SUCCESS,
                _contents=csv_data)
        else:
            table_data = dd.read_table(path, blocksize=self.block_size)
            return CorpusLoaderStatusResult(
                _status=CorpusLoaderStatus.SUCCESS,
                _contents=table_data)

    def import_corpus(self, corpus_name: str, is_labeled: bool, is_normalized: bool)-> CorpusLoaderStatusResult:
        child_folder_index = 0

        if not is_labeled:
            child_folder_index = 1
        if is_normalized:
            child_folder_index = 2

        if corpus_name not in self.__corpus_keys:
            return CorpusLoaderStatusResult(
                _status=CorpusLoaderStatus.ERROR_CORPUS_IMPORTING,
                _contents=f"folders:{self.__corpus_keys},input_corpus_name: {corpus_name} ",
                _issue=ErrorType.MISS_MATCHING)
        corpus_directory = f'{self.__base_path}/{self.__corpus[corpus_name]["name"]}/{self.corpus_child_folders[child_folder_index]}'
        file_list = os.listdir(corpus_directory)

        for file_name in file_list:
            path = f"{corpus_directory}/{file_name}"
            self.open_file(path,file_name)

        # with open(f'{self.base_path}/{self.corpus[corpus_name]["name"]}/{self.corpus_child_folders[child_folder_index]}', encoding="utf-8", mode="r") as file:
        #     if data_type.JSON:
        #         return CorpusLoaderStatusResult(
        #             _status=CorpusLoaderStatus.ERROR_CORPUS_FORMAT,
        #             _contents="",
        #             _issue=ErrorType.DATA_ISSUE)
        #     elif data_type.CSV:
        #         ...
        #     else data_type.TXT:
        #         ...
