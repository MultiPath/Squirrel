from torchtext import data
from torchtext import datasets
from data_loader import *
from utils import Watcher

import argparse

parser = argparse.ArgumentParser(description='Build a vocabulary for Transformer.')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument("--vocab_file",   type=str, required=True)

parser.add_argument('--base', type=str, default='bpe', choices=['byte', 'char', 'bpe', 'word'])
parser.add_argument('--c2', action='store_true', help='(experimental) used for input the 2D-char box.')
parser.add_argument('--max_vocab_size', type=int, default=80000, help='max vocabulary size')
parser.add_argument('--share_embeddings', action='store_true', help='share embeddings between encoder and decoder')

parser.add_argument('--additional_special', type=str, nargs='*', help='additional tokens needs to be added to the vocabulary')
parser.add_argument('--dec_init', type=str, default=None)
parser.add_argument('--dec_eos',  type=str, default=None)
parser.add_argument('--enc_init', type=str, default=None)
parser.add_argument('--enc_eos',  type=str, default=None)


parser.add_argument('--word_shuffle', type=int, default=3, help='Special for AE: the maximum range for words can be shuffled.')
parser.add_argument('--word_dropout', type=float, default=0.1, help='Special for AE: the maximum range for words can be dropped.')
parser.add_argument('--word_blank',   type=float, default=0.2, help='Special for AE: the maximum range for words can be paded.')

args = parser.parse_args() 

# -- default setting -- #
data_path = args.data_path
tokenizer = lambda s: s.split() 
revserse_tokenizer = lambda ex: " ".join(ex)
sort_key = None
Field = Seuqence

if args.base == 'byte':
    tokenizer = str2byte
    revserse_tokenizer = byte2str

elif args.base == 'char':
    tokenizer = lambda s: list(s)
    revserse_tokenizer = lambda ex: "".join(ex)

# -- source / target field --- #
common_kwargs = {'batch_first': True, 'tokenize': tokenizer, 'reverse_tokenize': revserse_tokenizer, 
                'shuffle': args.word_shuffle, 'dropout': args.word_dropout, 'replace': args.word_blank}
TRG = Field(init_token=args.dec_init, eos_token=args.dec_eos, **common_kwargs)

if args.share_embeddings:
    SRC = TRG
else:
    SRC = Field(init_token=args.enc_init, eos_token=args.enc_eos, **common_kwargs)
print('build the vocabulary.')

# --- setup dataset (no lazy mode when building the vocab) --- #
train_data = ParallelDataset(path= data_path, lazy=False,
    exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)], task='mt')

print('setup the dataset.')
if not args.share_embeddings:
    SRC.additional_tokens = args.additional_special
    SRC.build_vocab(train_data, max_size=args.max_vocab_size)
TRG.additional_tokens = args.additional_special
TRG.build_vocab(train_data, max_size=args.max_vocab_size)

torch.save([SRC.vocab, TRG.vocab], args.vocab_file)
print('done. {}/{}'.format(len(SRC.vocab), len(TRG.vocab)))
sys.exit(1)