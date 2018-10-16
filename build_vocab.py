from torchtext import data
from torchtext import datasets
from data_loader import DataLoader
from utils import Watcher

import argparse

parser = argparse.ArgumentParser(description='Build a vocabulary for Transformer.')
parser.add_argument('--data_prefix', type=str, default='/private/home/jgu/data/')
parser.add_argument('--dataset',     type=str, default='iwslt', help='name of datasets')
parser.add_argument('--src', type=str, default='en', help='source language marker')
parser.add_argument('--trg', type=str, default='de', help='target language marker')
parser.add_argument('--base', type=str, default='bpe', choices=['byte', 'char', 'bpe', 'word'])
parser.add_argument('--c2', action='store_true', help='(experimental) used for input the 2D-char box.')
parser.add_argument('--max_vocab_size', type=int, default=80000, help='max vocabulary size')
parser.add_argument('--train_set', type=str, default=None,  help='which train set to use')
parser.add_argument('--dev_set', type=str, default=None,  help='which train set to use')
parser.add_argument('--test_set', type=str, default=None,  help='which train set to use')
parser.add_argument('--share_embeddings', action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--remove_dec_eos', action='store_true', help='possibly remove <eos> tokens in the decoder')
parser.add_argument('--remove_enc_eos', action='store_true', help='possibly remove <eos> tokens in the encoder')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--distributed", default=False, type=bool)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--vocab_file", type=str, default=None)
parser.add_argument('--word_shuffle', type=int, default=3, help='Special for AE: the maximum range for words can be shuffled.')
parser.add_argument('--word_dropout', type=float, default=0.1, help='Special for AE: the maximum range for words can be dropped.')
parser.add_argument('--word_blank', type=float, default=0.2, help='Special for AE: the maximum range for words can be paded.')

args = parser.parse_args() 
watcher = Watcher(rank=args.local_rank, log_path=None)
DataLoader(args, watcher, build_vocab=True, vocab_file=args.vocab_file)


