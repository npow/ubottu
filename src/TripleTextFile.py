import nltk.tokenize
import numpy as np
from picklable_itertools import iter_, chain
from fuel.datasets import Dataset

class TripleTextFile(Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, files, dictionary, unk_token='<UNK>', max_len=40):
        self.files = files
        self.dictionary = dictionary
        if unk_token not in dictionary:
            raise ValueError
        self.unk_token = unk_token
        self.max_len = max_len
        self.data = np.array([self.process_line(line) for line in self.open()])
        super(TripleTextFile, self).__init__()

    def open(self):
        return chain(*[iter_(open(f)) for f in self.files])

    def get_data(self, state=None, request=None):
        if request is not None:
            return self.data[request]
        line = next(state)
        return self.process_line(line)

    def process_line(self, line):
        L = line.split('\t')
        y, x1, x2 = int(L[0]), L[3], L[4]

        # map words to indices in lookup table
        x1 = [self.dictionary.get(word, self.dictionary[self.unk_token]) for word in self.tokenize(x1)]
        x2 = [self.dictionary.get(word, self.dictionary[self.unk_token]) for word in self.tokenize(x2)]

        # pad with zeros to longest sequence
        x1.extend([0]*(self.max_len-len(x1)))
        x2.extend([0]*(self.max_len-len(x2)))

        return [np.array(x1, dtype='int32'), np.array(x2, dtype='int32'), y]

    def tokenize(self, line):
        tokens = nltk.tokenize.word_tokenize(line)
        return tokens[:self.max_len]
