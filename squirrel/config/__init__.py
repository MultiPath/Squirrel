import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import gmtime, strftime

import torch

FILE_ROOT = Path(__file__).parent
_CONFIGS = {}


def register_config(name):
    def register_config_fn(fn):
        if name in _CONFIGS:
            raise ValueError('Cannot register configurations')
        if not callable(fn):
            raise ValueError('Config builds must be callable ({name})')
        _CONFIGS[name] = fn
        return fn

    return register_config_fn


# automatically import any Python files in the scores/ directory
for m in os.listdir(FILE_ROOT):
    if m.endswith(('.py', '.pyc')) and not m.startswith('_'):
        model_name = m[:m.find('.py')]
        if model_name not in sys.modules:
            importlib.import_module('.' + model_name, 'squirrel.config')


def get_config(name='Training', seed=19920206):
    parser = argparse.ArgumentParser(description=name)
    # basic arguments
    parser.add_argument(
        '--mode', type=str, default='train', help='train, test or data'
    )  # "data": preprocessing and save vocabulary
    parser.add_argument(
        '--seed', type=int, default=seed, help='seed for randomness')

    for config in _CONFIGS:
        parser = _CONFIGS[config](parser)

    args = setup_config(parser.parse_args())
    return args


def setup_config(args):

    # ======================= fix the arguments ==================== #

    # languages
    srcs = args.src.split(',')
    trgs = args.trg.split(',')
    if args.test_src is not None:
        test_srcs = args.test_src.split(',')
        test_trgs = args.test_trg.split(',')
    else:
        args.test_src = args.src
        args.test_trg = args.trg

        test_srcs = srcs
        test_trgs = trgs

    all_langs = list(set(srcs + trgs + test_srcs + test_trgs))
    args.__dict__.update({
        'srcs': srcs,
        'trgs': trgs,
        'test_srcs': test_srcs,
        'test_trgs': test_trgs,
        'all_langs': all_langs
    })

    # cuda
    args.device_id = args.local_rank
    args.device = "cuda:{}".format(args.device_id)

    if 'MASTER_PORT' in os.environ:
        args.master_port = os.environ['MASTER_PORT']

    # use SLURM: multi-node training --- # hacky
    if 'SLURM_PROCID' in os.environ:
        node_list = os.environ['SLURM_JOB_NODELIST']
        hostnames = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', node_list]).split()

        args.init_method = 'tcp://{host}:{port}'.format(
            host=hostnames[0].decode('utf-8'), port=args.master_port)
        args.local_rank = args.local_rank + int(
            os.environ['SLURM_PROCID']) * int(os.environ['WORLD_SIZE'])
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
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.local_rank)

    # set the model prefix
    running_time = strftime("%m.%d_%H.%M.%S.", gmtime())
    if args.prefix == '[time]':
        args.prefix = running_time
    else:
        args.prefix = running_time + args.prefix

    # model hyper-params:
    hparams = {}
    if args.params == 'james-iwslt':
        hparams = {
            'd_model': 278,
            'd_hidden': 507,
            'n_layers': 5,
            'n_heads': 2,
            'drop_ratio': 0.079,
            'warmup': 746
        }  # ~32
    args.__dict__.update(hparams)

    args.params = '_'.join([
        str(a)
        for a in [args.d_model, args.d_hidden, args.n_layers, args.n_heads]
    ])

    # TODO: come up with a better name
    # model name (need more updates!!)
    hp_str = (
        f".{args.dataset}_{args.params}_"
        f"{args.src}_{args.trg}_"
        f"{args.model}_"
        f"{'causal_' if args.causal_enc else ''}"
        f"{'rp_' if args.relative_pos else ''}"
        f"{'ins_' if args.insertable else ''}"
        f"{'wf_' if args.insert_mode == 'word_first' else ''}"
        f"{'lm{}_'.format(args.lm_steps) if args.lm_steps > 0 else ''}"
        f"{args.base}_"
        f"{args.label_smooth}_"
        f"{args.inter_size*args.batch_size*args.world_size*len(args.src.split(','))}_"
        f"{'M{}'.format(args.multi_width) if args.multi_width > 1 else ''}")

    if args.load_from != 'none':
        hp_str += 'from_' + args.load_from[:15]
        if 'iter' in args.load_from:
            hp_str += args.load_from[args.load_from.index('iter'):]

        if args.resume:
            hp_str += '-C'

    model_name = os.path.join(args.workspace_prefix, 'models',
                              args.prefix + hp_str)
    args.__dict__.update({'model_name': model_name, 'hp_str': hp_str})

    # load arguments if provided
    if args.json is not None:
        saved_args = json.load(
            open(os.path.join(args.workspace_prefix, 'settings', args.json)))
        saved_args['local_rank'] = args.local_rank
        saved_args['prefix'] = args.prefix
        args.__dict__.update(saved_args)

    else:
        if args.local_rank == 0:
            with open(
                    os.path.join(args.workspace_prefix, 'settings',
                                 args.prefix + hp_str + '.json'),
                    'w') as outfile:
                json.dump(vars(args), outfile)

    return args


def setup_dataloader(args):
    if args.dataloader is not None:
        return args.dataloader
    else:
        if len(args.srcs) > 1:
            return 'multi'
        elif args.insertable:
            return 'order'
        else:
            return 'default'


def setup_datapath(args):
    # clean up the workspace
    if args.local_rank == 0:
        if not os.path.exists(args.workspace_prefix):
            os.mkdir(args.workspace_prefix)

        for d in ['models', 'runs', 'logs', 'decodes', 'settings']:
            if not os.path.exists(os.path.join(args.workspace_prefix, d)):
                os.mkdir(os.path.join(args.workspace_prefix, d))

    if args.decoding_path is None:
        decoding_path = os.path.join(
            args.workspace_prefix, 'decodes',
            args.load_from if args.mode == 'test' else
            (args.prefix + args.hp_str))
        if args.force_translate_from is not None:
            decoding_path = decoding_path + '_from_{}'.format(
                args.force_translate_from)
        if args.force_translate_to is not None:
            decoding_path = decoding_path + '_to_{}'.format(
                args.force_translate_to)

        args.decoding_path = decoding_path


def setup_pretrained_model(args, model, watcher=None):

    # load pre-trained parameters
    if args.load_from != 'none':
        checkpoint_file = os.path.join(args.workspace_prefix, 'models',
                                       args.load_from + '.pt')

        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                'No such checkpoint...{}'.format(checkpoint_file))

        with torch.cuda.device(args.device_id):
            pretrained_dict = torch.load(
                checkpoint_file,
                map_location=lambda storage, loc: storage.cuda())
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in model_dict
            }

            # discard some parameters which is not needed (optional)
            if args.never_load is not None:
                keys = []
                for k in pretrained_dict:
                    flag = 0
                    for s in args.never_load:
                        if s in k:
                            flag = 1
                            break
                    if flag == 0:
                        keys.append(k)
                pretrained_dict = {k: pretrained_dict[k] for k in keys}

            # for k in pretrained_dict:
            #     if args.local_rank == 0:
            #         watcher.info('load from checkpoints... {}'.format(k))

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            # force to freeze part of the parameters
            if args.freeze is not None:
                for n, p in model.named_parameters():
                    for fp in args.freeze:
                        if fp in n:
                            p.requires_grad = False
                            break

    return model
