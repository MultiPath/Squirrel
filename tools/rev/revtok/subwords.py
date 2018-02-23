# coding=utf-8
from .tokenizer import tokenize

from collections import defaultdict, Counter
from operator import attrgetter

from tqdm import tqdm

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret

class Utterance:
    def __init__(self, text):
        self.text = text
        self.count = 0
        self.ngrams = set()
    def overlaps(self, ngram1, ngram2):
        #print(self.text, ngram1.text, ngram2.text)
        inds1, inds2 = ngram1.utterances[self], ngram2.utterances[self]
        ret = 0
        for i1 in inds1:
            for i2 in inds2:
                #TODO verify all these exactly
                if i2 <= i1 <= i1 + ngram1.n <= i2 + ngram2.n:
                    ret += 1
                elif i1 <= i2 <= i2 + ngram2.n <= i1 + ngram1.n:
                    ret += 0
                elif i1 <= i2 < i1 + ngram1.n:
                    ret += 1 # - (i2 - i1) / ngram1.n
                elif i2 <= i1 < i2 + ngram2.n:
                    ret += 1 # (i1 - i2) / ngram1.n
        return ret / (len(inds1) * len(inds2))

class NGram:
    def __init__(self, text):
        self.n = len(text)
        self.text = text
        self.utterances = defaultdict(set)
        self._count = 0
    @property
    def count(self):
        return self._count
    @count.setter
    def count(self, value):
        self._count = value
        self.entropy = self._count * (self.n - 1)
    def add(self, utterance, i):
        self.count += utterance.count
        self.utterances[utterance].add(i)
        utterance.ngrams.add(self)
    def __repr__(self):
        return "'{0}': {1}".format(self.text, self.count)

class NGrams:
    def __init__(self, counter):
        self.ngrams = keydefaultdict(NGram)
        utterances = keydefaultdict(Utterance)
        for text, count in counter.items():
            utterances[text].count = count
        for utterance in tqdm(utterances.values(), 'enumerating ngrams'):
            self.from_utterance(utterance)
    def from_utterance(self, utterance):
        N = len(utterance.text)
        for i in range(N - 1):
            for n in range(2, N + 1 - i):
                self.ngrams[utterance.text[i:i+n]].add(utterance, i)

class SubwordSegmenter:
    # TODO MAYBE allow segmentations like " aware " + "ness "
    def __init__(self, counter, max_size):
        self.vocab = Counter(''.join(counter.keys())).most_common()
        self.vocab.sort(key=lambda tup: (-tup[1], tup[0]))
        self.vocab = dict(self.vocab)
        ngrams = list(NGrams(counter).ngrams.values())
        ngrams.sort(key=attrgetter('text'))
        key = attrgetter('entropy')
        for i in tqdm(range(max_size - len(self.vocab)), 'building vocab'):
            ngrams.sort(key=key, reverse=True)
            best = ngrams[0]
            #print(best)
            for utterance in best.utterances:
                seen = set([best])
                for ngram in utterance.ngrams:
                    if ngram not in seen:
                        ngram.count -= utterance.count * utterance.overlaps(ngram, best)
                        seen.add(ngram)
            self.vocab[ngrams[0].text] = ngrams[0].entropy
            ngrams = ngrams[1:]

    def __call__(self, utterance):
        #print(utterance)
        if utterance in self.vocab:
            return [utterance]
        i, segments = 0, {0: []}
        while True:
            for j in range(i + 1, len(utterance) + 1):
                if utterance[i:j] in self.vocab:
                    #print(i, j, segments)
                    curlen = len(segments[j]) if j in segments else len(utterance) + 1
                    if len(segments[i]) + 1 < curlen:
                        segments[j] = segments[i] + [utterance[i:j]]
            #print(i, segments)
            inds = sorted(segments.keys())
            if inds.index(i) < len(inds) - 1:
                i = inds[inds.index(i) + 1]
            else:
                break
        return segments[len(utterance)]

class SubwordTokenizer:
    def __init__(self, text, max_size):
        corpus = tokenize(text, decap=True)
        self.segmenter = SubwordSegmenter(Counter(corpus), max_size)
    def __call__(self, text):
        segments = map(self.segmenter, tokenize(text))
        return [tok for word in segments for tok in word]

# #corpus = ['megabyte', 'gigabyte']
# train = tokenize("""
# """)
# test = tokenize("""
# """)
# vocab = build_vocab(train, 1000)
# print(vocab)
# segments = [segment(tok, vocab) for tok in tqdm(test, 'segmenting')]
# print(segments)
# segments = [tok for word in segments for tok in word]
# print(len(segments))
