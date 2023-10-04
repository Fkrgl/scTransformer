import json
import sys
import numpy as np

class GeneVocab:
    """
    Vocabulary for genes
    """
    def __int__(self):
        self.vocab: dict
        self.pad_token: str = "<pad>"

    def init(self):
        """
        create dict with only a pad token
        """
        self.vocab = {}
        self.vocab["<pad>"] = 0

    def extend(self, tokens):
        """
        includes new tokens in the dict
        """
        next_id = self.get_last_id() + 1
        idx = next_id
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1

    def get_last_id(self) -> int:
        ids = list(sorted(self.vocab.values()))
        return ids[-1]

    def get_tokens(self):
        return list(self.vocab.keys())

    def get_token_ids(self):
        return list(self.vocab.values())

    def save_to_file(self, path_vocab) -> None:
        with open(path_vocab, 'w') as f:
            # write the dictionary to the file in JSON format
            json.dump(self.vocab, f)

    def load_from_file(self, path_file):
        """
        load an existing, at least initialized vocabulary from a json file
        """
        with open(path_file, 'r') as f:
            self.vocab = json.load(f)

# if __name__ == '__main__':
#     token_path = sys.argv[1]
#     out_vocab_path = sys.argv[2]
#     token = np.load(token_path, allow_pickle=True)
#     print(token)
#     v = GeneVocab()
#     v.init()
#     v.extend(token)
#     print(v.vocab)
#     v.save_to_file(out_vocab_path)

