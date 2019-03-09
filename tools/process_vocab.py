import torchtext
from torchtext import data
from torchtext import datasets
import copy
import torch
import os
from collections import Counter, OrderedDict
import argparse

parser = argparse.ArgumentParser(description='Build a vocabulary for Transformer.')
parser.add_argument('--data_path', type=str, default='./')

parser.add_argument("--vocab_file", type=str, default='es-de-fr.s.w.pt')
parser.add_argument('--src_train', type=str, default='es,de,fr', help='source language marker')
parser.add_argument('--trg_train', type=str, default='en,en,en', help='target language marker')

parser.add_argument('--base', type=str, default='bpe', choices=['byte', 'char', 'bpe', 'word'])
parser.add_argument('--max_vocab_size', type=int, default=80000, help='max vocabulary size')
parser.add_argument('--share_embeddings', action='store_false', help='share embeddings between encoder and decoder')
parser.add_argument('--eos',  type=str, default='<eos>')

args = parser.parse_args() 

# -- default setting -- #
data_path = args.data_path
tokenizer = lambda s: s.split() 

# -- source / target field --- #
srcs_train = args.src_train.split(',')
trgs_train = args.trg_train.split(',')
languages = list(set(srcs_train + trgs_train))
print('build the vocabulary.')

# --- setup dataset (no lazy mode when building the vocab) --- #
counter = Counter()
sources = []

for src, trg in zip([srcs_train, trgs_train]):
    DATADIR = '{}/{}-{}'.format(data_path, src, trg)
    if not os.path.exists(DATADIR):
        DATADIR = '{}/{}-{}'.format(data_path, trg, src)
        if not os.path.exists(DATADIR):
            raise FileNotFoundError
    with open('{}/train.bpe.src'.format(DATADIR), 'r') as f:
        lines = f.readlines()
        sources.append(lines)
    with open('{}/train.bpe.trg'.format(DATADIR), 'r') as f:
        lines = f.readlines()
        sources.append(lines)

for data in sources:
    for x in data:
        counter.update(tokenizer(x))

specials = list(OrderedDict.fromkeys(
    tok for tok in ['<unk>', '<pad>', '<init>', '<eos>'] + ['<{}>'.format(a) for a in languages] if tok is not None))
meta_vocab = torchtext.vocab.Vocab(counter, specials=specials, max_size=args.max_vocab_size)

print('setup the dataset.')

torch.save([meta_vocab, meta_vocab], data_path + '/' + args.vocab_file)
print('done. {}'.format(len(meta_vocab)))
print(meta_vocab.itos[:10])

