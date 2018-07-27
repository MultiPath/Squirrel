import torch
import numpy as np
from torchtext import data
from torchtext import datasets

import logging
import random
import argparse
import sys
import os
import copy

from ez_train import train_model
from ez_decode import decode_model
from model import Transformer, INF, TINY, softmax
from utils import NormalField, LazyParallelDataset, ParallelDataset, merge_cache
from time import gmtime, strftime


# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer-Like Model.')

# dataset settings --- 
parser.add_argument('--data_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--workspace_prefix', type=str, default='./') 

parser.add_argument('--dataset',     type=str, default='iwslt', help='"flickr" or "iwslt"')
parser.add_argument('--char', action='store_true', help='if --char enabled, character-based model are used.')
parser.add_argument('--src', type=str, default='en', help='source language marker')
parser.add_argument('--trg', type=str, default='de', help='target language marker')

parser.add_argument('--max_len',      type=int, default=None,  help='limit the train set sentences to this many tokens')
parser.add_argument('--max_vocab_size', type=int, default=50000, help='max vocabulary size')
parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--load_lazy', action='store_true', help='load a lazy-mode dataset, not save everything in the mem')
parser.add_argument('--remove_dec_eos', action='store_true', help='possibly remove <eos> tokens in the decoder')
parser.add_argument('--remove_enc_eos', action='store_true', help='possibly remove <eos> tokens in the encoder')

parser.add_argument('--train_set', type=str, default=None,  help='which train set to use')
parser.add_argument('--dev_set', type=str, default=None,  help='which dev set to use')
parser.add_argument('--test_set', type=str, default=None,  help='which test set to use')


# model basic settings
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='customize', help='pamarater sets: james-iwslt, t2t-base')

# customize
parser.add_argument('--d_model',  type=int, default=512,   help='basic parameter of the model size')
parser.add_argument('--d_hidden', type=int, default=2048,  help='used in feedforward network')
parser.add_argument('--warmup',   type=int, default=16000, help='warming-up steps during training')
parser.add_argument('--n_layers', type=int, default=6,     help='number of encoder-decoder')
parser.add_argument('--n_heads',  type=int, default=8,     help='number of heads for multi-head attention')
parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')

# model ablation settings
parser.add_argument('--causal_enc', action='store_true', help='use unidirectional encoder (useful for real-time translation)')
parser.add_argument('--encoder_lm', action='store_true', help='use unidirectional encoder with additional loss as a LM')
parser.add_argument('--causal',   action='store_true', help='use causal attention')
parser.add_argument('--cross_attn_fashion', type=str, default='forward', choices=['forward', 'reverse', 'last_layer'])
parser.add_argument('--share_embeddings',     action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--positional_attention', action='store_true', help='incorporate positional information in key/value')
parser.add_argument('--pry_io', action='store_true', help='(optional) multi-step prediction')
parser.add_argument('--pry_depth', type=int, default=1, help='deconv depth used in pry_io')

# running setting
parser.add_argument('--mode',    type=str, default='train',  help='train, test or data')  # "data": preprocessing and save vocabulary
parser.add_argument('--gpu',     type=int, default=0,        help='GPU to use or -1 for CPU')
parser.add_argument('--seed',    type=int, default=19920206, help='seed for randomness')

# training
parser.add_argument('--label_smooth',  type=float, default=0.1,   help='regularization via label-smoothing during training.')
parser.add_argument('--eval_every',    type=int, default=1000,    help='run dev every')
parser.add_argument('--save_every',    type=int, default=50000,   help='save the best checkpoint every 50k updates')
parser.add_argument('--maximum_steps', type=int, default=1000000, help='maximum steps you take to train a model')
parser.add_argument('--inter_size',    type=int, default=4,       help='process multiple batches before one update')
parser.add_argument('--batch_size',    type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer-style learning rate')

# decoding
parser.add_argument('--length_ratio',  type=int,   default=2, help='maximum lengths of decoding')
parser.add_argument('--beam_size',     type=int,   default=1, help='beam-size used in Beamsearch, default using greedy decoding')
parser.add_argument('--alpha',         type=float, default=1, help='length normalization weights')
parser.add_argument('--no_bpe', action='store_true', help='do not output BPE in the decoding mode.')

# model saving/reloading, output translations
parser.add_argument('--load_from',     type=str, default=None, help='load from checkpoint')
parser.add_argument('--resume',        action='store_true', help='when loading from the saved model, it resumes from that.')

# debugging
parser.add_argument('--debug',       action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')

# arguments (:)
args = parser.parse_args()

if not os.path.exists(args.workspace_prefix):
    os.mkdir(args.workspace_prefix)

for d in ['models', 'runs', 'logs', 'decodes']:    # check the path
    if not os.path.exists(os.path.join(args.workspace_prefix, d)):
        os.mkdir(os.path.join(args.workspace_prefix, d))
if args.prefix == '[time]':
    args.prefix = strftime("%m.%d_%H.%M.", gmtime())

# setup logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

fh = logging.FileHandler(os.path.join(args.workspace_prefix, 'logs', 'log-{}.txt'.format(args.prefix)))
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

# special for Pytorch 0.4
args.device = "cuda:{}".format(args.gpu) if args.gpu > -1 else "cpu"

# setup a data-field
DataField = NormalField

if not args.char:
    tokenizer = lambda s: s.split() 
else:
    tokenizer = lambda s: list(s)
    
if args.remove_dec_eos:
    TRG = DataField(batch_first=True, tokenize=tokenizer)
else:
    TRG = DataField(init_token='<init>', eos_token='<eos>', batch_first=True, tokenize=tokenizer)

if args.share_embeddings:
    SRC = TRG
elif args.remove_enc_eos:
    SRC = DataField(batch_first=True, tokenize=tokenizer)
else:
    SRC = DataField(init_token='<init>', eos_token='<eos>', batch_first=True, tokenize=tokenizer)
    

# read the dataset
DatasetFunc = LazyParallelDataset if (args.load_lazy and args.mode != 'data') else ParallelDataset
train_data, dev_data, test_data = DatasetFunc.splits(
    path=os.path.join(args.data_prefix, args.dataset, args.src + '-' + args.trg) + '/', 
    train=args.train_set, validation=args.dev_set, test=args.test_set, 
    exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)])
vocab_name = 'vocab.{}-{}.{}.{}.pt'.format(args.src, args.trg, 
                                        's' if args.share_embeddings else 'n',
                                        'c' if args.char else 'w')

if args.mode == 'data':

    # build vocabulary
    if not args.share_embeddings:
        SRC.build_vocab(train_data, dev_data, max_size=args.max_vocab_size)
    TRG.build_vocab(train_data, dev_data, max_size=args.max_vocab_size)
    
    torch.save([SRC.vocab, TRG.vocab], os.path.join(args.data_prefix, args.dataset, args.src + '-' + args.trg, vocab_name))
    logger.info('save the processed vocabulary, {} {}'.format(len(SRC.vocab), len(TRG.vocab)))
    sys.exit(1)

else:

    # load vocabulary
    assert os.path.exists(os.path.join(args.data_prefix, args.dataset, args.src + '-' + args.trg, vocab_name)), 'need to pre-compute the vocab'

    logger.info('load saved vocabulary.')
    src_vocab, trg_vocab = torch.load(os.path.join(args.data_prefix, args.dataset, args.src + '-' + args.trg, vocab_name))

    SRC.vocab = src_vocab
    TRG.vocab = trg_vocab

args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

# dynamic batching
def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.src), len(new.trg),  prev_max_len) * i

def dyn_batch_with_overhead(new, i, sofar):
    
    def oh(x):
        return x * (1 + 0.001 * x)

    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(oh(len(new.src)), oh(len(new.trg)),  prev_max_len) * i

def dyn_batch_without_padding(new, i, sofar):
    return sofar + max(len(new.src), len(new.trg))


if args.batch_size == 1:  # speed-test: one sentence per batch.
    batch_size_fn = lambda new, count, sofar: count
else:
    if not args.char:
        #batch_size_fn = dyn_batch_without_padding
        batch_size_fn = dyn_batch_with_padding
        
    else:
        batch_size_fn = dyn_batch_with_overhead


# batch-iterator
train_real, dev_real = data.BucketIterator.splits(
    (train_data, dev_data), batch_sizes=(args.batch_size, args.batch_size), 
    device=args.device, batch_size_fn=batch_size_fn, 
    repeat=None if args.mode == 'train' else False,
    shuffle=(not args.load_lazy))
logger.info("build the dataset. done!")
# ----------------------------------------------------------------------------------------------------------------- #

# model hyper-params:
hparams = None
if args.params == 'james-iwslt':
    hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
elif args.param == 't2t-base':
    logger.info('use default parameters of t2t-base')  # t2t-base, 512-2048-6
    hparams = {'d_model': 512, 'd_hidden': 2048, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
else:
    logger.info("following the user setting.")


args.__dict__.update(hparams)


hp_str = (f"{args.dataset}_{args.params}_"
          f"{args.src}_{args.trg}_"
          f"{'causal_' if args.causal_enc else ''}"
          f"{'lm_' if args.encoder_lm else ''}"
          f"{'c' if args.char else 'w'}_"
          f"{args.label_smooth}_"
          f"{args.inter_size*args.batch_size}")
logger.info(f'Starting with HPARAMS: {hp_str}')
model_name = os.path.join(args.workspace_prefix, 'models', args.prefix + hp_str)

# build the model
model = Transformer(SRC, TRG, args)
logger.info(str(model))

# use GPU 
if args.gpu > -1:
    model.to(torch.device(args.device))

# load pre-trained parameters
if args.load_from is not None:
    with torch.cuda.device(args.gpu):
        model.load_state_dict(torch.load(
            os.path.join(args.workspace_prefix, 'models', args.load_from + '.pt'),
            map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.


# additional information
args.__dict__.update({'model_name': model_name, 'hp_str': hp_str,  'logger': logger})

# show
args_str = ''
for a in args.__dict__:
    args_str += '{}:\t{}\n'.format(a, args.__dict__[a])
logger.info(args_str)


# ----------------------------------------------------------------------------------------------------------------- #
if args.mode == 'train':
    logger.info('starting training')
    train_model(args, model, train_real, dev_real)

elif args.mode == 'test':
    logger.info('starting decoding from the pre-trained model, on the test set...')
    assert args.load_from is not None, 'must decode from a pre-trained model.'

    decoding_path = os.path.join(args.workspace_prefix, 'decodes', args.load_from)
    if not os.path.exists(decoding_path):
        os.mkdir(decoding_path)
    name_suffix = 'b={}_a={}.txt'.format(args.beam_size, args.alpha)
    names = ['{}.src.{}'.format(args.test_set, name_suffix), 
             '{}.trg.{}'.format(args.test_set, name_suffix),
             '{}.dec.{}'.format(args.test_set, name_suffix)]
    with torch.no_grad():   
        decode_model(args, model, dev_real, evaluate=True, decoding_path=decoding_path, names=names)

logger.info("done.")
