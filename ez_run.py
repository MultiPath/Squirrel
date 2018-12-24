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
import subprocess

from torchtext import data
from torchtext import datasets
from time import gmtime, strftime

from learner import train_model, train_2phases
from decoder import valid_model

from models.core import INF, TINY, softmax
from models.transformer_ins import TransformerIns
from models.transformer_iww import TransformerIww
from models.transformer_sink import TransformerSink
from models.transformer import Transformer
from models.transformer_vae import AutoTransformer, AutoTransformer2, AutoTransformer3

from data_loader import MultiDataLoader, OrderDataLoader
from utils import *
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer-Like Model.')

# dataset settings --- 
parser.add_argument('--data_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--workspace_prefix', type=str, default='./') 
parser.add_argument('--dataset',     type=str, default='iwslt', help='"flickr" or "iwslt"')
parser.add_argument('--src', type=str, default='en', help='source language marker')
parser.add_argument('--trg', type=str, default='de', help='target language marker')
parser.add_argument('--test_src', type=str, default=None, help='source language marker in testin')
parser.add_argument('--test_trg', type=str, default=None, help='target language marker')
parser.add_argument('--vocab_file', type=str, default=None, help='user-defined vocabulary')
parser.add_argument('--lang_as_init_token', action='store_true', help='use language token as initial tokens')

parser.add_argument('--max_vocab_size', type=int, default=50000, help='max vocabulary size')
parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--load_lazy', action='store_true', help='load a lazy-mode dataset, not save everything in the mem')
parser.add_argument('--remove_dec_eos', action='store_true', help='possibly remove <eos> tokens in the decoder')
parser.add_argument('--remove_enc_eos', action='store_true', help='possibly remove <eos> tokens in the encoder')
parser.add_argument('--train_set', type=str, default=None,  help='which train set to use')
parser.add_argument('--dev_set', type=str, default=None,  help='which dev set to use')
parser.add_argument('--test_set', type=str, default=None,  help='which test set to use')

# insertable transformer
parser.add_argument('--insertable', action='store_true')
parser.add_argument('--insert_mode', choices=['word_first', 'position_first', 'balanced'], type=str, default='word_first')
parser.add_argument('--order', choices=['fixed', 'random', 'optimal', 'search_optimal', 'trainable'], type=str, default='fixed')
parser.add_argument('--path_temp', default=0, type=float, help='temperature to choose paths. 0 means choosing the top path only')
parser.add_argument('--beta', type=int, default=4, help='beam-size to search optimal paths.')
parser.add_argument('--ln_pos', action='store_true', help='a linear layer over the embedding and query.')
parser.add_argument('--no_bound', action='store_true', help='no boundary probabilitis.')
parser.add_argument('--no_weights', action='store_true', help='do not use reweighting after beam-search.')
parser.add_argument('--search_with_dropout', action='store_true', help='no boundary probabilitis.')

parser.add_argument('--epsilon', type=float, default=0, help='possibility to choose random order during training.')
parser.add_argument('--gamma', type=float, default=1, help='balance p(x) and p(z)')
parser.add_argument('--sample_order', action='store_true', help='perform sampling instead of beam-search')
parser.add_argument('--resampling', action='store_true', help='resampling after every samling operations')
parser.add_argument('--adaptive_ess_ratio', default=0.375, type=float, help='th of adaptive effective sample size')
parser.add_argument('--use_gumbel', action='store_true', help='resampling after every samling operations')
parser.add_argument('--esteps', type=int, default=1, help='possibility to choose random order during training.')
parser.add_argument('--gsteps', type=int, default=1, help='possibility to choose random order during training.')
parser.add_argument('--decouple', action='store_true', help='decouple the scorer and the trainer. use best model.')


# multi-lingual training
parser.add_argument('--multi', action='store_true', help='enable multilingual training for Transformer.')
parser.add_argument('--sample_prob', nargs='*', type=float, help='probabilities of each input dataset.')
parser.add_argument('--local_attention', type=int, default=0, help='force to use local attention for the first K layers.')

# character/byte-level Transformer
parser.add_argument('--base', type=str, default='bpe', choices=['byte', 'char', 'bpe', 'word'])
parser.add_argument('--c2', action='store_true', help='(experimental) used for input the 2D-char box.')

# (variational) auto-encoder settings
parser.add_argument('--autoencoding', action='store_true', help='Train autoencoder')
parser.add_argument('--ae_func', type=str, default='all_steps', choices=['first_step', 'all_steps'])
parser.add_argument('--n_proj_layers', type=int, default=6)

# denoising auto-encoder settings
parser.add_argument('--pool', type=str, default='mean', choices=['mean', 'max'], help='pooling used to extract sentence information.')
parser.add_argument('--input_noise', type=str, default=None, choices=['n1', 'n2', 'n3'], help='inject input space noise (for Denoising Auto-Encoder)')
parser.add_argument('--word_shuffle', type=int, default=3, help='Special for AE: the maximum range for words can be shuffled.')
parser.add_argument('--word_dropout', type=float, default=0.1, help='Special for AE: the maximum range for words can be dropped.')
parser.add_argument('--word_blank', type=float, default=0.2, help='Special for AE: the maximum range for words can be paded.')
parser.add_argument('--latent_noise', type=float, default=0, help='inject latent space noise which is a Gaussian (for Denoising Auto-Encoder)')

# model basic settings
parser.add_argument('--model',  type=str, default='Transformer', choices=['Transformer', 'TransformerIns', 'TransformerIww', 'AutoTransformer', 'AutoTransformer2', 'AutoTransformer3', 'TransformerSink'])
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='customize', help='pamarater sets: james-iwslt, t2t-base')

# customize
parser.add_argument('--d_model',  type=int, default=512,   help='basic parameter of the model size')
parser.add_argument('--d_hidden', type=int, default=2048,  help='used in feedforward network')
parser.add_argument('--warmup',   type=int, default=4000,  help='warming-up steps during training')
parser.add_argument('--n_layers', type=int, default=6,     help='number of encoder-decoder')
parser.add_argument('--n_heads',  type=int, default=8,     help='number of heads for multi-head attention')
parser.add_argument('--n_cross_heads', type=int, default=8,  help='number of heads for multi-head attention')
parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio for attention')
parser.add_argument('--relu_drop_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--attn_drop_ratio', type=float, default=0.0, help='dropout ratio')

# model ablation settings
parser.add_argument('--block_order', type=str, default='tdan', choices=['tdan', 'tdna', 'tnda'])
parser.add_argument('--normalize_emb', action='store_true', help='normalize embedding (IO)')
parser.add_argument('--causal_enc', action='store_true', help='use unidirectional encoder (useful for real-time translation)')
parser.add_argument('--encoder_lm', action='store_true', help='use unidirectional encoder with additional loss as a LM')
parser.add_argument('--causal',   action='store_true', help='use causal attention')
parser.add_argument('--cross_attn_fashion', type=str, default='forward', choices=['forward', 'reverse', 'last_layer'])
parser.add_argument('--share_embeddings', action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--uniform_embedding_init', action='store_true', help='by default, we use Transformer clever init for embeddings. But we can always go back to pytorch default.')
parser.add_argument('--relative_pos', action='store_true', help="""
                                                                use relative position in the attention, instead of positional encoding.
                                                                currently supports the simplest case: left (0), self(1), right(2)
                                                                """)

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
parser.add_argument('--print_every',   type=int, default=0,       help='print every during training')
parser.add_argument('--att_plot_every',type=int, default=0,       help='visualization the attention matrix of a sampled training set.')
parser.add_argument('--save_every',    type=int, default=50000,   help='save the best checkpoint every 50k updates')
parser.add_argument('--maximum_steps', type=int, default=1000000, help='maximum steps you take to train a model')

parser.add_argument('--lm_steps',      type=int, default=0,       help='pre-training steps without encoder inputs.')
parser.add_argument('--lm_schedule',   action='store_true',       help='instead of switching LM and MT, we incorporate a scheduling, gradually increasing the prop.')

parser.add_argument('--sub_inter_size',type=int, default=1,       help='process multiple batches before one update')
parser.add_argument('--inter_size',    type=int, default=4,       help='process multiple batches before one update')
parser.add_argument('--batch_size',    type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--valid_batch_size', type=int, default=2048, help='# of tokens processed per batch')
parser.add_argument('--maxlen',        type=int, default=10000,   help='limit the train set sentences to this many tokens')
parser.add_argument('--maxatt_size',   type=int, default=2200000, help= """
                                                                        limit the maximum attention computation in order to avoid OOM.
                                                                        Dynamic batching makes sure: #sent x #token <= batch-size
                                                                        Dynamic batching makes sure: #sent x #token ^ 2 <= maxatt-size
                                                                        """)
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--lr',            type=float, default=0)
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer-style learning rate')
parser.add_argument('--weight_decay',  type=float, default=0)
parser.add_argument('--grad_clip',     type=float, default=25)

# decoding
parser.add_argument('--length_ratio',  type=int,   default=3, help='maximum lengths of decoding')
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
parser.add_argument('--valid_ppl',   action='store_true', help='debug mode: validation with ppl')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')


'''
Add some distributed options. For explanation of dist-url and dist-backend please see
http://pytorch.org/tutorials/intermediate/dist_tuto.html
--local_rank will be supplied by the Pytorch launcher wrapper (torch.distributed.launch)
'''
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--device_id", default=0, type=int)
parser.add_argument("--distributed", default=False, type=bool)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--init_method", default=None, type=str)
parser.add_argument("--master_port", default=11111, type=int)

# load pre_saved arguments
parser.add_argument("--json", default=None, type=str)

# arguments (:)
args = parser.parse_args() 
args.device_id  = args.local_rank
args.device = "cuda:{}".format(args.device_id) 

if 'MASTER_PORT' in os.environ:
    args.master_port = os.environ['MASTER_PORT']

# use SLURM: multi-node training --- # hacky
if 'SLURM_PROCID' in os.environ:
    node_list = os.environ['SLURM_JOB_NODELIST']
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list]).split()

    args.init_method = 'tcp://{host}:{port}'.format(host=hostnames[0].decode('utf-8'), port=args.master_port) 
    args.local_rank = args.local_rank + int(os.environ['SLURM_PROCID']) * int(os.environ['WORLD_SIZE'])
    args.world_size = int(os.environ['WORLD_SIZE']) * len(hostnames)
    args.distributed = True

else:
    # single node training
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1 
        args.init_method = 'tcp://localhost:{}'.format(args.master_port)

# setup multi-gpu
torch.cuda.set_device(args.device_id)
if args.distributed:
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=args.local_rank)

if args.local_rank == 0:
    if not os.path.exists(args.workspace_prefix):
        os.mkdir(args.workspace_prefix)

    for d in ['models', 'runs', 'logs', 'decodes', 'settings']:    # check the path
        if not os.path.exists(os.path.join(args.workspace_prefix, d)):
            os.mkdir(os.path.join(args.workspace_prefix, d))

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
                'n_heads': 8, 'n_cross_heads': 8, 'drop_ratio': 0.1, 'warmup': 4000}  # ~32
else:
    pass

if args.model == 'AutoTransformer2':
    args.n_cross_heads = 1

args.__dict__.update(hparams)

# model name
hp_str = (  f".{args.dataset}_{args.params}_"
            f"{args.src}_{args.trg}_"
            f"{args.model}_"
            f"{'causal_' if args.causal_enc else ''}"
            f"{'rp_' if args.relative_pos else ''}"
            f"{'ins_' if args.insertable else ''}"
            f"{'wf_' if args.insert_mode == 'word_first' else ''}"
            f"{'lm{}_'.format(args.lm_steps) if args.lm_steps > 0 else ''}"
            f"{args.base}_"
            f"{args.label_smooth}_"
            f"{args.inter_size*args.batch_size*args.world_size}_"
            f"{'M{}'.format(args.multi_width) if args.multi_width > 1 else ''}"
        )

if args.load_from != 'none':
    hp_str += 'from_' + args.load_from
    if args.resume:
        hp_str += '-C'

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

# setup random seeds
setup_random_seed(args.seed)

# setup watcher settings
watcher = Watcher(rank=args.local_rank, log_path=os.path.join(args.workspace_prefix, 'logs', 'log-{}.txt'.format(args.prefix)))
watcher.info('\n'.join(['{}:\t{}'.format(a, b) for a, b in sorted(args.__dict__.items(), key=lambda x: x[0])]))
watcher.info(f'Starting with HPARAMS: {hp_str}')

# ========================================================================================================== #

# get the dataloader
if args.insertable:
    dataloader = OrderDataLoader(args, watcher, vocab_file=args.vocab_file)
else:
    dataloader = MultiDataLoader(args, watcher, vocab_file=args.vocab_file)

model = eval(args.model)(dataloader.SRC, dataloader.TRG, args)  # build the model either Transformer or AutoEncoder.
watcher.info(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
watcher.info("total trainable parameters: {}".format(format(count_parameters(model),',')))
watcher.info("Vocabulary size: {}/{}.".format(len(dataloader.SRC.vocab), len(dataloader.TRG.vocab)))

# use GPU 
if torch.cuda.is_available():
    model.cuda()

if args.distributed:
    model = DDP(model, device_ids=[args.device_id], output_device=args.device_id)

print("RANK:{}, WORLD_SIZE:{}, DEVICE-ID:{}, MASTER={}".format(args.local_rank, args.world_size, args.init_method, args.device_id))

# load pre-trained parameters
if args.load_from != 'none':
    with torch.cuda.device(args.device_id):
        pretrained_dict = torch.load(
            os.path.join(args.workspace_prefix, 'models', args.load_from + '.pt'),
            map_location=lambda storage, loc: storage.cuda())
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

decoding_path = os.path.join(args.workspace_prefix, 'decodes', args.load_from if args.mode == 'test' else (args.prefix + hp_str))
if (args.local_rank == 0) and (not os.path.exists(decoding_path)):
    os.mkdir(decoding_path)

# start running
if args.mode == 'train':
    watcher.info('starting training')
    train_model(args, watcher, model, dataloader.train, dataloader.dev, decoding_path=decoding_path)

    # if args.autoencoding:  # running auto-encoder
    #     train_autoencoder(args, watcher, model, dataloader.train, dataloader.dev)
    # else:
    
    # if 'AutoTransformer' in args.model:
    #     train_2phases(args, watcher, model, dataloader.train, dataloader.dev, decoding_path=decoding_path)
    # else:

elif args.mode == 'test':
    watcher.info('starting decoding from the pre-trained model, on the test set...')
    assert args.load_from is not None, 'must decode from a pre-trained model.'

    with torch.no_grad(): 
        test_set = dataloader.test if args.decode_test else dataloader.dev
        name = '{}.b={}_a={}.txt'.format(args.test_set if args.decode_test else args.dev_set, args.beam_size, args.alpha)
        decoding_path += '/{}'.format(name)

        for set_i in test_set:
            if args.autoencoding: # evaluating auto-encoder
                valid_model(args, watcher, model, set_i, decoding_path=decoding_path, dataflow=['src', 'src'])
                valid_model(args, watcher, model, set_i, decoding_path=decoding_path, dataflow=['trg', 'trg'])
            else:
                valid_model(args, watcher, model, set_i, print_out=True, decoding_path=decoding_path, dataflow=['src', 'trg'])

watcher.info("done.")
