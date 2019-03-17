import datetime
import os
import subprocess
import time

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from squirrel.config import (get_config, setup_dataloader, setup_datapath,
                             setup_pretrained_model)
from squirrel.data import get_dataloader
from squirrel.decoder import valid_model, valid_model_ppl
from squirrel.learner import train_model
from squirrel.models import get_model
from squirrel.utils import Watcher, count_parameters, setup_random_seed


# all the hyper-parameters
def master():
    args = get_config("Transformer-Squirrel")
    setup_random_seed(args.seed)
    setup_datapath(args)
    cudnn.deterministic = True

    # cuda
    if 'MASTER_PORT' in os.environ:
        args.master_port = os.environ['MASTER_PORT']

    # # use SLURM: multi-node training --- # hacky
    if 'SLURM_PROCID' in os.environ:
        node_list = os.environ['SLURM_JOB_NODELIST']
        hostnames = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', node_list]).split()

        args.init_method = 'tcp://{host}:{port}'.format(
            host=hostnames[0].decode('utf-8'), port=args.master_port)

        args.world_size = len(hostnames)
        args.local_rank = int(os.environ['SLURM_PROCID'])
        args.distributed = True

    else:
        # single node training
        if 'WORLD_SIZE' in os.environ:
            args.world_size = int(os.environ['WORLD_SIZE'])

        args.distributed = True  # right now we only consider distributed.
        args.init_method = 'tcp://localhost:{}'.format(args.master_port)

    mp.spawn(
        worker, nprocs=args.nproc_per_node, args=(args.nproc_per_node, args))

    # # setup multi-gpu
    # torch.cuda.set_device(args.device_id)
    # if args.distributed:
    #     torch.distributed.init_process_group(
    #         backend='nccl',
    #         init_method=args.init_method,
    #         world_size=args.world_size,
    #         rank=args.local_rank)


def worker(gpu, npgpus_per_node, args):

    START_TIME = time.time()

    args.device_id = gpu
    args.device = "cuda:{}".format(gpu)
    args.local_rank = args.local_rank * npgpus_per_node + gpu
    args.world_size = args.world_size * npgpus_per_node

    if args.distributed:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.local_rank)

    torch.cuda.set_device(args.device_id)

    # setup watcher settings
    watcher = Watcher(
        rank=args.local_rank,
        log_path=os.path.join(args.workspace_prefix, 'logs',
                              'log-{}.txt'.format(args.prefix)) if
        ((args.logfile is None) or (args.logfile == 'none')) else args.logfile)
    watcher.info('\n'.join([
        '{}:\t{}'.format(a, b)
        for a, b in sorted(args.__dict__.items(), key=lambda x: x[0])
    ]))
    watcher.info('Starting with HPARAMS: {}'.format(args.hp_str))
    watcher.info("RANK:{}, WORLD_SIZE:{}, DEVICE-ID:{}, MASTER={}".format(
        args.local_rank, args.world_size, args.init_method, args.device_id))

    # get the dataloader
    dataloader = get_dataloader(setup_dataloader(args))(
        args, watcher, vocab_file=args.vocab_file)

    # get the model
    model = get_model(args.model)(dataloader.SRC, dataloader.TRG, args)
    watcher.info(model)
    watcher.info("total trainable parameters: {}".format(
        format(count_parameters(model), ',')))
    watcher.info("Vocabulary size: {}/{}.".format(
        len(dataloader.SRC.vocab), len(dataloader.TRG.vocab)))

    # use GPU
    if torch.cuda.is_available():
        model.cuda()
    if args.distributed:
        model = DDP(
            model, device_ids=[args.device_id], output_device=args.device_id)
    model = setup_pretrained_model(args, model, watcher)

    # start running
    if args.mode == 'train':
        watcher.info('starting training')
        train_model(
            args,
            watcher,
            model,
            dataloader.train,
            dataloader.dev,
            decoding_path=None)

    elif args.mode == 'test':

        if (args.local_rank == 0) and (not os.path.exists(args.decoding_path)):
            os.mkdir(args.decoding_path)

        watcher.info(
            'starting decoding from the pre-trained model, on the test set...')
        assert args.load_from is not None, 'must decode from a pre-trained model.'

        with torch.no_grad():
            test_set = dataloader.test if args.decode_test else dataloader.dev
            name = '{}.b={}_a={}.txt'.format(
                args.test_set if args.decode_test else args.dev_set,
                args.beam_size, args.alpha)
            args.decoding_path += '/{}'.format(name)

            for set_i in test_set:
                valid_model(
                    args,
                    watcher,
                    model,
                    set_i,
                    print_out=True,
                    decoding_path=args.decoding_path,
                    dataflow=['src', 'trg'])

    elif args.mode == 'valid_ppl':

        watcher.info(
            'starting to evaluate the model from the pre-trained model, on the test set...'
        )
        assert args.load_from is not None, 'must decode from a pre-trained model.'

        with torch.no_grad():
            test_set = dataloader.test if args.decode_test else dataloader.dev
            for set_i in test_set:
                if args.sweep_target_tokens is not None:
                    target_tokens = [
                        '<{}>'.format(a)
                        for a in args.sweep_target_tokens.split(',')
                    ]
                else:
                    target_tokens = [set_i.init_tokens['trg']]

                for trg_tok in target_tokens:
                    set_i.init_tokens['trg'] = trg_tok
                    watcher.info("{} -> {}".format(set_i.task,
                                                   set_i.init_tokens))
                    output_file = open(
                        args.decoding_path + '/{}->{}.txt'.format(
                            set_i.task, set_i.init_tokens['trg'][1:-1]), 'w')
                    outputs = valid_model_ppl(
                        args,
                        watcher,
                        model,
                        set_i,
                        dataflow=['src', 'trg'],
                        lm_only=args.lm_only)

                    if args.local_rank == 0:
                        for s, t, ppl in zip(*[
                                outputs['src'], outputs['trg'], outputs['loss']
                        ]):
                            line = '{}\t{}\t{}'.format(ppl, s, t)
                            print(line, file=output_file, flush=True)
                        print('write done.')

    watcher.info("all done.  Total clock time = {}".format(
        str(datetime.timedelta(seconds=(time.time() - START_TIME)))))


if __name__ == '__main__':
    master()
