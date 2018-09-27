import torch
import numpy as np
import logging
import random
import argparse
import sys
import os
import copy
import json
import torch.distributed as dist

from torchtext import data
from torchtext import datasets
from time import gmtime, strftime

from trainer import train_model, valid_model
from core import INF, TINY, softmax
from transformer import Transformer
from data_loader import DataLoader
from utils import Watcher
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer-Like Model.')

# dataset settings --- 
parser.add_argument('--data_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--workspace_prefix', type=str, default='./') 

parser.add_argument('--dataset',     type=str, default='iwslt', help='"flickr" or "iwslt"')
parser.add_argument('--src', type=str, default='en', help='source language marker')
parser.add_argument('--trg', type=str, default='de', help='target language marker')
parser.add_argument('--vocab_file', type=str, default=None, help='user-defined vocabulary')

# character/byte-level Transformer
parser.add_argument('--base', type=str, default='bpe', choices=['byte', 'char', 'bpe', 'word'])
parser.add_argument('--c2', action='store_true', help='(experimental) used for input the 2D-char box.')

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
parser.add_argument('--exp',    type=str, default='transformer', help='useless')
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='customize', help='pamarater sets: james-iwslt, t2t-base')

# customize
parser.add_argument('--d_model',  type=int, default=512,   help='basic parameter of the model size')
parser.add_argument('--d_hidden', type=int, default=2048,  help='used in feedforward network')
parser.add_argument('--warmup',   type=int, default=4000,  help='warming-up steps during training')
parser.add_argument('--n_layers', type=int, default=6,     help='number of encoder-decoder')
parser.add_argument('--n_heads',  type=int, default=8,     help='number of heads for multi-head attention')
parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')

# model ablation settings
parser.add_argument('--block_order', type=str, default='tdan', choices=['tdan', 'tdna', 'tnda'])
parser.add_argument('--normalize_emb', action='store_true', help='normalize embedding (IO)')
parser.add_argument('--causal_enc', action='store_true', help='use unidirectional encoder (useful for real-time translation)')
parser.add_argument('--encoder_lm', action='store_true', help='use unidirectional encoder with additional loss as a LM')
parser.add_argument('--causal',   action='store_true', help='use causal attention')
parser.add_argument('--cross_attn_fashion', type=str, default='forward', choices=['forward', 'reverse', 'last_layer'])
parser.add_argument('--share_embeddings',     action='store_true', help='share embeddings between encoder and decoder')

# MS-decoder: blockwise parallel decoding 
parser.add_argument('--multi_width', type=int, default=1, help='default not use multi-step prediction')
parser.add_argument('--dyn', type=float, default=0.0, help='dynamic block-wse decoding (experimental)')
parser.add_argument('--random_path', action='store_true', help='use a random path, instead of dynamic block-wse decoding (experimental)')
parser.add_argument('--exact_match', action='store_true', help='match with the 1-step model in dynamic block-wse decoding (experimental)')
parser.add_argument('--constant_penalty', type=float, default=0)


# running setting
parser.add_argument('--mode',    type=str, default='train',  help='train, test or data')  # "data": preprocessing and save vocabulary
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
parser.add_argument('--original',      action='store_true', help='output the original output files, not the tokenized ones.')
parser.add_argument('--decode_test',   action='store_true', help='evaluate scores on test set instead of using dev set.')

# Learning to Translation in Real-Time
parser.add_argument('--real_time', action='store_true', help='real-time translation.')


# model saving/reloading, output translations
parser.add_argument('--load_from', type=str, default='none', help='load from checkpoint')
parser.add_argument('--resume',    action='store_true', help='when loading from the saved model, it resumes from that.')

# debugging
parser.add_argument('--debug',       action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--no_valid',    action='store_true', help='debug mode: no validation')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')


'''
Add some distributed options. For explanation of dist-url and dist-backend please see
http://pytorch.org/tutorials/intermediate/dist_tuto.html
--local_rank will be supplied by the Pytorch launcher wrapper (torch.distributed.launch)
'''
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--distributed", default=False, type=bool)
parser.add_argument("--world_size", default=1, type=int)

# load pre_saved arguments
parser.add_argument("--json", default=None, type=str)

# arguments (:)
args = parser.parse_args() 
if args.local_rank == 0:
    if not os.path.exists(args.workspace_prefix):
        os.mkdir(args.workspace_prefix)

    for d in ['models', 'runs', 'logs', 'decodes', 'settings']:    # check the path
        if not os.path.exists(os.path.join(args.workspace_prefix, d)):
            os.mkdir(os.path.join(args.workspace_prefix, d))


if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.distributed = args.world_size > 1

running_time = strftime("%m.%d_%H.%M.%S.", gmtime())
if args.prefix == '[time]':
    args.prefix = running_time
else:
    args.prefix = running_time + args.prefix

# model hyper-params:
hparams = {}
if args.params == 'james-iwslt':
    hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
elif args.params == 't2t-base':
    hparams = {'d_model': 512, 'd_hidden': 2048, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 4000}  # ~32
else:
    pass

args.__dict__.update(hparams)

# model name
hp_str = (  f".{args.dataset}_{args.params}_"
            f"{args.src}_{args.trg}_"
            f"{'causal_' if args.causal_enc else ''}"
            f"{'lm_' if args.encoder_lm else ''}"
            f"{args.base}_"
            f"{args.label_smooth}_"
            f"{args.inter_size*args.batch_size*args.world_size}_"
            f"{'M{}'.format(args.multi_width)}"
        )


model_name = os.path.join(args.workspace_prefix, 'models', args.prefix + hp_str)
args.__dict__.update({'model_name': model_name, 'hp_str': hp_str})

# load arguments if provided
if args.json is not None:
    saved_args = json.load(open(os.path.join(args.workspace_prefix, 'settings', args.json)))
    saved_args['local_rank'] = args.local_rank
    saved_args['prefix'] = args.prefix

    args.__dict__.update(saved_args)

else:
    if args.local_rank == 0:
        with open(os.path.join(args.workspace_prefix, 'settings', args.prefix + hp_str + '.json'), 'w') as outfile:
            json.dump(vars(args), outfile)

# ========================================================================================= #

# special for Pytorch 0.4
args.device = "cuda:{}".format(args.local_rank) 

# setup multi-gpu
torch.cuda.set_device(args.local_rank)
if args.distributed:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

# setup random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# setup watcher settings
watcher = Watcher(rank=args.local_rank, log_path=os.path.join(args.workspace_prefix, 'logs', 'log-{}.txt'.format(args.prefix)))
watcher.info('\n'.join(['{}:\t{}'.format(a, b) for a, b in sorted(args.__dict__.items(), key=lambda x: x[0])]))
watcher.info(f'Starting with HPARAMS: {hp_str}')

# ========================================================================================================== #

# get the dataloader
dataloader = DataLoader(args, watcher, vocab_file=args.vocab_file)

# build the model
model = Transformer(dataloader.SRC, dataloader.TRG, args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
watcher.info("total trainable parameters: {}".format(format(count_parameters(model),',')))
watcher.info("Vocabulary size: {}/{}.".format(len(dataloader.SRC.vocab), len(dataloader.TRG.vocab)))

# use GPU 
if torch.cuda.is_available():
    model.cuda()

if args.distributed:
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

# load pre-trained parameters
if args.load_from != 'none':
    with torch.cuda.device(args.local_rank):
        model.load_state_dict(torch.load(
            os.path.join(args.workspace_prefix, 'models', args.load_from + '.pt'),
            map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.


decoding_path = os.path.join(args.workspace_prefix, 'decodes', args.load_from if args.mode == 'test' else (args.prefix + hp_str))
if (args.local_rank == 0) and (not os.path.exists(decoding_path)):
    os.mkdir(decoding_path)

name_suffix = 'b={}_a={}.txt'.format(args.beam_size, args.alpha)
names = ['{}.src.{}'.format(args.test_set, name_suffix), '{}.trg.{}'.format(args.test_set, name_suffix), '{}.dec.{}'.format(args.test_set, name_suffix)]

# start running
if args.mode == 'train':
    watcher.info('starting training')
    train_model(args, watcher, model, dataloader.train, dataloader.dev, decoding_path=decoding_path, names=names)

elif args.mode == 'test':

    watcher.info('starting decoding from the pre-trained model, on the test set...')
    assert args.load_from is not None, 'must decode from a pre-trained model.'
    with torch.no_grad():   
        valid_model(args, watcher, model, dataloader.test if args.decode_test else dataloader.dev, decoding_path=decoding_path, names=names)

watcher.info("done.")
