from tokenizer.vocab import Vocab

test_dict = {"ㄱ":1, "ㄴ": 2, "ㄷ": 3, "<pad>": 4, "<unk>": 5, "<mask>": 6}

def test_vocab():
    sentence = "ㄱㄴㄷㄹ"

    encoded_contents = [1,2,3,5,4,4,4]
    vocab = Vocab(test_dict, padding_token="<pad>", unknown_token="<unk>")

    print("vocab.get_id(char)")
    for char in sentence:
        print(vocab.get_id(char))

    print("vocab.get_token(id)")
    for id in encoded_contents:
        print(vocab.get_token(id))


    print("vocab.get_vocab()")
    print(vocab.get_vocab())

if __name__ == '__main__':
    test_vocab()