from collections import Counter
from typing import Iterable
from saf import Sentence


class Vocab(object):
    def __init__(self, corpus: Iterable[Sentence], size: int):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [w.surface for s in corpus for w in s.tokens]
        self.cnt = Counter(words)
        n_unk = len(words)
        for w, c in self.cnt.most_common(size):
            v.append(w)
            n_unk -= c
        self.cnt['<unk>'] = n_unk

        self.word2idx = {}
        self.idx2word = []

        for w in v:
            self.word2idx[w] = len(self.word2idx)
            self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']


