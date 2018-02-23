import torch
import numpy as np
from torchtext import data
from torchtext import datasets
from torch.nn import functional as F
from torch.autograd import Variable

import revtok
import logging
import random
import string
import traceback
import math
import uuid
import argparse
import os
import copy
import time

from train import train_model
from decode import decode_model
from model import Transformer, FastTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, merge_cache
from xliffdata import XLIFFDataset
from time import gmtime, strftime

import sys
from traceback import extract_tb
from code import interact

# check paths
for d in ['models', 'runs', 'logs', 'self_play']:
    if not os.path.exists('./{}'.format(d)):
        os.mkdir('./{}'.format(d))

# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer / NAT.')

# dataset settings
parser.add_argument('--data_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--dataset',     type=str, default='iwslt', help='"flickr" or "iwslt"')
parser.add_argument('--language',    type=str, default='ende',  help='a combination of two language markers to show the language pair.')
parser.add_argument('--data_group',  type=str, default=None,  help='dataset group')

parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--load_dataset', action='store_true', help='load a pre-processed dataset')
parser.add_argument('--use_revtok',   action='store_true', help='use reversible tokenization')
parser.add_argument('--remove_eos',   action='store_true', help='possibly remove <eos> tokens for FastTransformer')
parser.add_argument('--test_set',     type=str, default=None,  help='which test set to use')
parser.add_argument('--max_len',      type=int, default=None,  help='limit the train set sentences to this many tokens')

# model basic settings
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='james-iwslt', help='pamarater sets: james-iwslt, t2t-base, etc')

# model ablation settings
parser.add_argument('--local',    dest='windows', action='store_const', const=[1, 3, 5, 7, -1], default=None, help='use local attention')
parser.add_argument('--causal',   action='store_true', help='use causal attention')
parser.add_argument('--diag',     action='store_true', help='ignore diagonal attention when doing self-attention.')
parser.add_argument('--use_wo',   action='store_true', help='use output weight matrix in multihead attention')

parser.add_argument('--fertility',            action='store_true', help='use the latent fertility model. only useful for NAT')
parser.add_argument('--hard_inputs',          action='store_true', help='use hard selection as inputs, instead of soft-attention over embeddings.')
parser.add_argument('--use_alignment',        action='store_true', help='use the aligned fake data to initialize')
parser.add_argument('--input_orderless',      action='store_true', help='for the inputs, remove the order information')
parser.add_argument('--share_embeddings',     action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--supervise_fertility',  action='store_true', help='directly use the groud-truth alignment for reordering.')
parser.add_argument('--positional_attention', action='store_true', help='incorporate positional information in key/value')

# running setting
parser.add_argument('--mode',    type=str, default='train',  help='train, test or build')
parser.add_argument('--gpu',     type=int, default=0,        help='GPU to use or -1 for CPU')
parser.add_argument('--seed',    type=int, default=19920206, help='seed for randomness')

# training
parser.add_argument('--eval-every',    type=int, default=1000,    help='run dev every')
parser.add_argument('--save_every',    type=int, default=50000,   help='save the best checkpoint every 50k updates')
parser.add_argument('--maximum_steps', type=int, default=1000000, help='maximum steps you take to train a model')
parser.add_argument('--batch_size',    type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer-style learning rate')

parser.add_argument('--distillation', action='store_true', help='knowledge distillation at sequence level')
parser.add_argument('--finetuning',   action='store_true', help='knowledge distillation at word level')

# decoding
parser.add_argument('--length_ratio',  type=int,   default=2, help='maximum lengths of decoding')
parser.add_argument('--decode_mode',   type=str,   default='argmax', help='decoding mode: argmax, mean, sample, noisy, search')
parser.add_argument('--beam_size',     type=int,   default=1, help='beam-size used in Beamsearch, default using greedy decoding')
parser.add_argument('--f_size',        type=int,   default=1, help='heap size for sampling/searching in the fertility space')
parser.add_argument('--alpha',         type=float, default=1, help='length normalization weights')
parser.add_argument('--temperature',   type=float, default=1, help='smoothing temperature for noisy decodig')
parser.add_argument('--rerank_by_bleu', action='store_true', help='use the teacher model for reranking')

# self-playing
parser.add_argument('--max_cache',    type=int, default=20,   help='save most recent max_cache sets of translations')
parser.add_argument('--decode_every', type=int, default=2000, help='every 1k updates, train the teacher again')
parser.add_argument('--train_every',  type=int, default=500,  help='train the teacher again for 250k steps')

# model saving/reloading, output translations
parser.add_argument('--load_from',     type=str, default=None, help='load from checkpoint')
parser.add_argument('--resume',        action='store_true', help='when loading from the saved model, it resumes from that.')
parser.add_argument('--share_encoder', action='store_true', help='use teacher-encoder to initialize student')

parser.add_argument('--no_bpe',        action='store_true', help='output files without BPE')
parser.add_argument('--no_write',      action='store_true', help='do not write the decoding into the decoding files.')
parser.add_argument('--output_fer',    action='store_true', help='decoding and output fertilities')

# other settings:

parser.add_argument('--real_data', action='store_true', help='only used in the reverse kl setting')
parser.add_argument('--beta1', type=float, default=0.5, help='balancing MLE and KL loss.')
parser.add_argument('--beta2', type=float, default=0.01, help='balancing the GAN loss.')

# debugging
parser.add_argument('--debug',       action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')

# (maybe not useful any more) old params
parser.add_argument('--old', action='store_true', help='this is used for solving conflicts of new codes')
# ----------------------------------------------------------------------------------------------------------------- #

args = parser.parse_args()
if args.prefix == '[time]':
    args.prefix = strftime("%m.%d_%H.%M.", gmtime())

# get the langauage pairs:
args.src = args.language[:2]  # source language
args.trg = args.language[2:]  # target language

# setup logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

fh = logging.FileHandler('./logs/log-{}.txt'.format(args.prefix))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# setup random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ----------------------------------------------------------------------------------------------------------------- #
# setup data-field
DataField = data.ReversibleField if args.use_revtok else NormalField
TRG   = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
SRC   = DataField(batch_first=True) if not args.share_embeddings else TRG

# setup many datasets (need to manaually setup)
data_prefix = args.data_prefix
if args.dataset == 'iwslt':
    if args.test_set is None:
        args.test_set = 'IWSLT16.TED.tst2013'

    if args.data_group == 'test':
        train_data, dev_data = NormalTranslationDataset.splits(
        path=data_prefix + 'iwslt/en-de/', train='train.tags.en-de.bpe',
        validation='{}.en-de.bpe'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
        fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='normal')

    else:  # default dataset
        train_data, dev_data = ParallelDataset.splits(
        path=data_prefix + 'iwslt/en-de/', train='train.tags.en-de.bpe',
        validation='train.tags.en-de.bpe.dev', exts=('.en2', '.de2'),
        fields=[('src', SRC), ('trg', TRG)],
        load_dataset=args.load_dataset, prefix='ts')

    decoding_path = data_prefix + 'iwslt/en-de/{}.en-de.bpe.new'

else:
    raise NotImplementedError

# build vocabularies
if args.load_vocab and os.path.exists(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg))):

    logger.info('load saved vocabulary.')
    src_vocab, trg_vocab = torch.load(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg)))
    SRC.vocab = src_vocab
    TRG.vocab = trg_vocab

else:

    logger.info('save the vocabulary')
    if not args.share_embeddings:
        SRC.build_vocab(train_data, dev_data, max_size=50000)
    TRG.build_vocab(train_data, dev_data, max_size=50000)
    torch.save([SRC.vocab, TRG.vocab], data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg)))
args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

# build alignments ---
def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    if args.distillation:
        return max(len(new.src), len(new.trg), len(new.dec), prev_max_len) * i
    else:
        return max(len(new.src), len(new.trg),  prev_max_len) * i

def dyn_batch_without_padding(new, i, sofar):
    if args.distillation:
        return sofar + max(len(new.src), len(new.trg), len(new.dec))
    else:
        return sofar + max(len(new.src), len(new.trg))

# work around torchtext making it hard to share vocabs without sharing other field properties
if args.share_embeddings:
    SRC = copy.deepcopy(SRC)
    SRC.init_token = None
    SRC.eos_token = None
    train_data.fields['src'] = SRC
    dev_data.fields['src'] = SRC

if args.max_len is not None:
    train_data.examples = [ex for ex in train_data.examples if len(ex.trg) <= args.max_len]

if args.batch_size == 1:  # speed-test: one sentence per batch.
    batch_size_fn = lambda new, count, sofar: count
else:
    batch_size_fn = dyn_batch_without_padding

train_real, dev_real = data.BucketIterator.splits(
    (train_data, dev_data), batch_sizes=(args.batch_size, args.batch_size), device=args.gpu,
    batch_size_fn=batch_size_fn, repeat=None if args.mode == 'train' else False)
logger.info("build the dataset. done!")
# ----------------------------------------------------------------------------------------------------------------- #

# model hyper-params:
hparams = None
if args.dataset == 'iwslt':
    if args.params == 'james-iwslt':
        hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
    elif args.params == 'james-iwslt2':
        hparams = {'d_model': 278, 'd_hidden': 2048, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
    teacher_hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746}


if hparams is None:
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
args.__dict__.update(hparams)

# ----------------------------------------------------------------------------------------------------------------- #
# show the arg:
logger.info(args)

hp_str = (f"{args.dataset}_subword_"
        f"{args.d_model}_{args.d_hidden}_{args.n_layers}_{args.n_heads}_"
        f"{args.drop_ratio:.3f}_{args.warmup}_"
        f"{args.xe_until if hasattr(args, 'xe_until') else ''}_"
        f"{f'{args.xe_ratio:.3f}' if hasattr(args, 'xe_ratio') else ''}_"
        f"{args.xe_every if hasattr(args, 'xe_every') else ''}")
logger.info(f'Starting with HPARAMS: {hp_str}')
model_name = './models/' + args.prefix + hp_str

# build the model
model = Transformer(SRC, TRG, args, causal_enc=True)
logger.info(str(model))
if args.load_from is not None:
    with torch.cuda.device(args.gpu):
        model.load_state_dict(torch.load('./models/' + args.load_from + '.pt',
        map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.

# use cuda
if args.gpu > -1:
    model.cuda(args.gpu)

# additional information
args.__dict__.update({'model_name': model_name, 'hp_str': hp_str,  'logger': logger})

# ----------------------------------------------------------------------------------------------------------------- #
if args.mode == 'train':
    logger.info('starting training')
    train_model(args, model, train_real, dev_real)

elif args.mode == 'test':
    logger.info('starting decoding from the pre-trained model, on the test set...')
    name_suffix = '{}_b={}_model_{}.txt'.format(args.decode_mode, args.beam_size, args.load_from)
    names = ['src.{}'.format(name_suffix), 'trg.{}'.format(name_suffix),'dec.{}'.format(name_suffix)]

    if args.model is FastTransformer:
        names += ['fer.{}'.format(name_suffix)]
    if args.rerank_by_bleu:
        teacher_model = None
    decode_model(args, model, dev_real, evaluate=True, decoding_path=decoding_path if not args.no_write else None, names=names)

logger.info("done.")
