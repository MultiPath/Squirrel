from translate.storage import xliff
from torchtext import data
from torchtext.datasets import TranslationDataset

from nltk.tokenize import sent_tokenize
from nltk.translate import gale_church
import tqdm

import os
import glob

def multilingual_sent_tokenize(text):
    if '。' in text:
        return text.split('。')
        # TODO FIXME
    return sent_tokenize(text)

def sent_align(src, trg):
    src, trg = map(multilingual_sent_tokenize, (src, trg))
    if len(src) == len(trg) == 1:
        return zip(src, trg)
    lens = gale_church.align_texts([[len(s) for s in src]], [[len(s) for s in trg]])
    i, j = -1, -1
    examples = []
    for pair in lens[0]:
        if pair[0] != i and pair[1] != j:
            examples.append([[], []])
        if pair[0] != i:
            examples[-1][0].append(src[pair[0]])
        if pair[1] != j:
            examples[-1][1].append(trg[pair[1]])
        i, j = pair
    return [(' '.join(x[0]), ' '.join(x[1])) for x in examples]

class XLIFFDataset(TranslationDataset):
    folder = '/home/james.bradbury/Downloads/sfdc_mt_data/fi/'

    def __init__(self, paths, fields, **kwargs):
        examples = []
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
        for fname in tqdm.tqdm(paths, desc='loading XLIFF files'):
            with open(fname) as f:
                xliff_file = xliff.xlifffile.parsestring(f)
            for unit in xliff_file.units:
                src, trg = unit.source, unit.target
                if (src is not None and trg is not None and
                        min(len(src), len(trg)) > 0):
                    examples.extend(sent_align(src, trg))

        examples = [data.Example.fromlist(x, fields) for x in tqdm.tqdm(
            set(examples), desc='tokenizing and preprocessing')]
        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, train=True, validation=True, test=True,
               ext='.xml.sdlxliff', **kwargs):
        path = os.path.expanduser(path)
        paths = glob.glob(path + '*' + ext)
        train_paths = paths[:int(0.96*len(paths))]
        val_paths = paths[int(0.96*len(paths)):int(0.98*len(paths))]
        test_paths = paths[int(0.98*len(paths)):]
        train_data = None if not train else cls(train_paths, **kwargs)
        val_data = None if not validation else cls(val_paths, **kwargs)
        test_data = None if not test else cls(test_paths, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


if __name__ == '__main__':
    path = '/home/james.bradbury/Downloads/sfdc_mt_data/fi/'
    SRC = data.Field(tokenize='moses')
    TRG = data.Field(tokenize='moses')
    train, dev, test = XLIFFDataset.splits(path, fields=(SRC, TRG))
