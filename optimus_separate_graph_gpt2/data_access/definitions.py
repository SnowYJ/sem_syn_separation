"""
Data loader for WorldTree, Wikipeida, WordNet, Wiktionary datasets.
"""
import sys
from typing import Iterable
from zipfile import ZipFile
from saf import Sentence, Token
import pandas as pd

DSR_FILES = {
    "wikipedia": "data/DSR/Wikipedia/WKP_DSR_model_CSV.zip",
    "wiktionary": "../data/DSR/Wikitionary/WKT_DSR_model_CSV.zip",
    "wordnet": "data/DSR/WordNet/WN_DSR_model_CSV.zip",
    "debug": "data/DSR/debug/debug_wn.zip"
}

ESR_FILES = {
    "entailmentbankREC": "../../dataset/ESR/EntailmentBank/entailmentbank.txt",
    "entailmentbankINF": "./datasets/full/entailmentbankINF.csv",
    "entailmentbankCON": "../data/ESR/EntailmentBank/entailmentbankCON.txt",
    "entailmentbankVQVAE": "./datasets/full/explanations_vqvae.txt"
}

# ---------------------------------------------------------------------------------------
# loading definition
class DefinitionSemanticRoleCorpus(Iterable[Sentence]):
    def __init__(self, path: str):
        if (path in DSR_FILES):
            path = DSR_FILES[path]

        dsr_zip = ZipFile(path)
        self._source = dsr_zip.open(dsr_zip.namelist()[0])

        self._size = 0

        for line in self._source:
            self._size += 1

        self._source.seek(0)

    def __iter__(self):
        return DefinitionSemanticRoleCorpusIterator(self)

    def __len__(self):
        return self._size


class DefinitionSemanticRoleCorpusIterator:
    def __init__(self, dsrc: DefinitionSemanticRoleCorpus):
        self._dsrc = dsrc
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        k = 0
        sentence_buffer = [None]
        while (sentence_buffer):
            sentence_buffer = list()
            while (not sentence_buffer):
                try:
                    line_bytes = next(self._dsrc._source)
                    line_bytes = line_bytes.replace(b"\r", b"\\r").replace(b"\t", b"\\t")
                    line_bytes = line_bytes.replace(b"\N", b"\\N").replace(b"\c", b"\\c")
                    line_bytes = line_bytes.replace(b"\i", b"\\i")
                    line = line_bytes.decode("unicode_escape")
                    line = line.strip().replace("&amp;", "&").replace("&quot;", "\"")
                    fields = line.split(";")
                    terms = fields[2].split(", ")


                    for term in terms:
                        sentence = Sentence()
                        sentence.annotations["id"] = fields[0]
                        sentence.annotations["POS"] = fields[1]
                        sentence.annotations["definiendum"] = term
                        sentence.annotations["definition"] = fields[3]

                        for i in range(4, len(fields)):
                            segment_role = fields[i].split("/")
                            segment = "/".join(segment_role[:-1])
                            role = segment_role[-1]
                            for tok in segment.split():
                                token = Token()
                                token.surface = tok
                                token.annotations["DSR"] = role
                                sentence.tokens.append(token)


                        sentence_buffer.append(sentence)

                except UnicodeDecodeError:
                    print("Decode error", file=sys.stderr)
                except StopIteration:
                    break

            for sentence in sentence_buffer:
                # print([t.surface for t in sentence.tokens])

                yield sentence

# ---------------------------------------------------------------------------------------
# loading Explanations
class ExplanationSemanticRoleCorpus(Iterable[Sentence]):
    def __init__(self, path: str):
        if (path in ESR_FILES):
            path = ESR_FILES[path]

        # dsr_zip = ZipFile(path)
        self._source = open(path)

        self._size = 0

        for line in self._source:
            self._size += 1

        self._source.seek(0)

    def __iter__(self):
        return ExplanationSemanticRoleCorpusIterator(self)

    def __len__(self):
        return self._size


class ExplanationSemanticRoleCorpusIterator:
    def __init__(self, dsrc: ExplanationSemanticRoleCorpus):
        self._dsrc = dsrc
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        k = 0
        sentence_buffer = [None]
        while (sentence_buffer):
            sentence_buffer = list()
            while (not sentence_buffer):
                try:
                    line = next(self._dsrc._source)
                    fields = line.split("&")
                    explain, semantic = fields[0], fields[1][:-1]

                    explain, semantic = explain.split(' '), semantic.split(' ')
                    explain, semantic = [i for i in explain if i != ''], [i for i in semantic if i != '']
                    assert len(explain) == len(semantic)

                    sentence = Sentence()

                    for tok, role in zip(explain, semantic):
                        token = Token()
                        token.surface = tok
                        token.annotations["DSR"] = role
                        sentence.tokens.append(token)

                    sentence_buffer.append(sentence)

                except UnicodeDecodeError:
                    print("Decode error", file=sys.stderr)
                except StopIteration:
                    break

            for sentence in sentence_buffer:
                yield sentence

# ---------------------------------------------------------------------------------------
# loading others read txt file each line is a sentence.
class LoadNormalCorpus(Iterable[Sentence]):
    def __init__(self, path: str):

        # dsr_zip = ZipFile(path)
        self._source = open(path)

        self._size = 0

        for line in self._source:
            self._size += 1

        self._source.seek(0)

    def __iter__(self):
        return LoadNormalCorpusIterator(self)

    def __len__(self):
        return self._size


class LoadNormalCorpusIterator:
    def __init__(self, dsrc: LoadNormalCorpus):
        self._dsrc = dsrc
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        k = 0
        sentence_buffer = [None]
        while (sentence_buffer):
            sentence_buffer = list()
            while (not sentence_buffer):
                try:
                    line = next(self._dsrc._source)[:-1] # remove \n
                    explain = line.split(' ')
                    sentence = Sentence()

                    for tok in explain:
                        token = Token()
                        token.surface = tok
                        sentence.tokens.append(token)

                    sentence_buffer.append(sentence)

                except UnicodeDecodeError:
                    print("Decode error", file=sys.stderr)
                except StopIteration:
                    break

            for sentence in sentence_buffer:
                yield sentence


def ExplanationInferenceCorpus(path):
    path = ESR_FILES[path]

    data = pd.read_csv(path, index_col=[0])
    max_length = 500
    examples = []

    for p1, p2, c in zip(data['premise1'], data['premise2'], data['conclusion']):
        if len(p1.split()) < 1 or len(p2.split()) < 1 or len(c.split()) < 1:
            continue

        if len(p1.split()) > max_length or len(p2.split()) > max_length or len(c.split()) > max_length:
            continue

        example = {
            'premises': [p1, p2],
            'conclusion': c
        }

        examples.append(example)

    return examples





if __name__ == "__main__":
    dsrc = DefinitionSemanticRoleCorpus("wordnet")
    print("Corpus size:", len(dsrc))

    i = 0
    for sent in dsrc:
        print("Sent annotations:", sent.annotations)
        print("Token annotations:", [(token.surface, token.annotations["DSR"]) for token in sent.tokens])
        i += 1
        if (i > 10):
           break
