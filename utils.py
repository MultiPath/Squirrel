import torch
import random
import torch.nn as nn
import numpy as np
import os
import torch.distributed as dist
import time
import pickle
import operator
import logging
import matplotlib
matplotlib.use('Agg')

import pylab as plt
import seaborn as sns

from torch.autograd import Variable
from torchtext import data, datasets
from torchtext.data.batch import Batch
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu

from contextlib import ExitStack
from collections import OrderedDict
from timeit import default_timer
from functools import reduce
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from termcolor import colored
from subprocess import PIPE, Popen
from tools.bleu_score import corpus_bleu


COLORS = ['red', 'green', 'yellow', 'blue', 'white', 'magenta', 'cyan']

INF = 1e10
TINY = 1e-9

params = {'legend.fontsize': 'x-large',
            'figure.figsize': (15, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large',
            'figure.max_open_warning': 200}
plt.rcParams.update(params)
sns.set()


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prod(factors):
    return reduce(operator.mul, factors, 1)

def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor

def format(x, type='k'):
    if type == 'k':
        return '{:.2f}K'.format(x / 1000)
    elif type == 'm':
        return '{:.3f}M'.format(x / 1000000)
    return x

def export(x):
    if isinstance(x, dict):
        for w in x:
            x[w] = export(x[w])
        return x

    try:
        with torch.cuda.device_of(x):
            return x.data.cpu().float().mean()
    except Exception:
        return 0

# detokonizer used when evaluating BLEU scores
def debpe(x):
    """
    de-byte pair encoding, return to tokenized text
    """
    return x.replace('@@ ', '').split()

def dechar(x):
    """
    de-tokonize the text to plain characters (no space)
    """
    return list(x.replace('@@ ', '').replace(' ', ''))

def seg_kytea(outputs):
    """
    special for Japanese text:
    :: resegment the text to segmented pieces with Kytea
    :: make sure to install kytea first ::
        http://www.phontron.com/kytea/
    """
    def _tok(x):
        return [xi.split('/')[0] for xi in x.split()]

    p = Popen('kytea', stdin=PIPE, stdout=PIPE, stderr=PIPE)
    outputs = ["" if len(o) == 0 else ''.join(o) for o in outputs]
    
    # very important to put \n in the end.
    return [_tok(o) for o in p.communicate(('\n'.join(outputs) + '\n').encode('utf-8'))[0].decode('utf-8').split('\n')]


def computeGLEU(outputs, targets, corpus=False, tokenizer=None, segmenter=None):
    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]

    if segmenter is not None:
        outputs = segmenter(outputs)
        targets = segmenter(targets)

    if not corpus:
        return [sentence_gleu([t],  o) for o, t in zip(outputs, targets)]
    return corpus_gleu([[t] for t in targets], [o for o in outputs])

def computeBLEU(outputs, targets, corpus=False, tokenizer=None, segmenter=None):
    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]

    if segmenter is not None:
        outputs = segmenter(outputs)
        targets = segmenter(targets)

    if not corpus:
        return torch.Tensor([sentence_gleu(
            [t],  o) for o, t in zip(outputs, targets)])
    return corpus_bleu([[t] for t in targets], [o for o in outputs], emulate_multibleu=True)

def computeGroupBLEU(outputs, targets, tokenizer=None, bra=10, maxmaxlen=80):
    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]
    maxlens = max([len(t) for t in targets])
    print(maxlens)
    maxlens = min([maxlens, maxmaxlen])
    nums = int(np.ceil(maxlens / bra))
    outputs_buckets = [[] for _ in range(nums)]
    targets_buckets = [[] for _ in range(nums)]
    for o, t in zip(outputs, targets):
        idx = len(o) // bra
        if idx >= len(outputs_buckets):
            idx = -1
        outputs_buckets[idx] += [o]
        targets_buckets[idx] += [t]

    for k in range(nums):
        print(corpus_bleu([[t] for t in targets_buckets[k]], [o for o in outputs_buckets[k]], emulate_multibleu=True))


def colored_seq(seq, bound, char=False):
    seq = seq.split() if not char else seq
    new_seq = []
    c_id = 0

    for k, w in enumerate(seq):
        if bound[k] == 0:
            c_id = (c_id + 1) % len(COLORS)

        new_seq.append(colored(w, COLORS[c_id]))
    
    if not char:
        return ' '.join(new_seq)
    else:
        return ''.join(new_seq)

def visualize_attention(seq1, seq2, attention):
    fig, ax = plt.subplots(figsize=(len(seq1) // 3, len(seq2) // 3), dpi=100)
    sns.heatmap(attention, ax=ax, cbar=False, cmap=sns.cubehelix_palette(start=2.4, rot=.1, light=1), square=True, xticklabels=seq1, yticklabels=seq2)
    ax.xaxis.tick_top()
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    return fig


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timer = default_timer
        
    def __enter__(self):
        self.start = self.timer()
        return self
        
    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.elapsed)


class Metrics:

    def __init__(self, name, *metrics):
        self.count = 0
        self.metrics = OrderedDict((metric, 0) for metric in metrics)
        self.name = name

    def accumulate(self, count, *values, print_iter=None):
        self.count += count
        if print_iter is not None:
            print(print_iter, end=' ')
        for value, metric in zip(values, self.metrics):
            if isinstance(value, torch.autograd.Variable):
                value = value.data
            if torch.is_tensor(value):
                with torch.cuda.device_of(value):
                    value = value.cpu()
                value = value.float().mean()

            if print_iter is not None:
                print('%.3f' % value, end=' ')
            self.metrics[metric] += value * count
        if print_iter is not None:
            print()
        return values[0] # loss

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key] / (self.count + 1e-9)
        raise AttributeError

    def __repr__(self):
        return (f"{self.name}: " +
                ', '.join(f'{metric}: {getattr(self, metric):.3f}'
                        for metric, value in self.metrics.items()
                        if value is not 0))

    def tensorboard(self, expt, i):
        for metric in self.metrics:
            value = getattr(self, metric)
            if value != 0:
                expt.add_scalar_value(f'{self.name}_{metric}', value, step=i)

    def reset(self):
        self.count = 0
        self.metrics.update({metric: 0 for metric in self.metrics})


class Best:

    def __init__(self, cmp_fn, *metrics, model=None, opt=None, path='', gpu=0):
        self.cmp_fn = cmp_fn
        self.model = model
        self.opt = opt
        self.path = path + '.pt'
        self.metrics = OrderedDict((metric, None) for metric in metrics)
        self.gpu = gpu

    def accumulate(self, cmp_value, *other_values):

        with torch.cuda.device(self.gpu):
            cmp_metric, best_cmp_value = list(self.metrics.items())[0]
            if best_cmp_value is None or self.cmp_fn(
                    best_cmp_value, cmp_value) == cmp_value:

                self.metrics[cmp_metric] = cmp_value
                self.metrics.update({metric: value for metric, value in zip(
                    list(self.metrics.keys())[1:], other_values)})

                open(self.path + '.temp', 'w')
                if self.model is not None:
                    torch.save(self.model.state_dict(), self.path)
                if self.opt is not None:
                    if isinstance(self.opt, list):
                        torch.save([self.i, [opt.state_dict() for opt in self.opt]], 
                                    self.path + '.states')
                    else:
                        torch.save([self.i, self.opt.state_dict()], self.path + '.states')
                os.remove(self.path + '.temp')

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key]
        raise AttributeError

    def __repr__(self):
        return ("BEST: " +
                ', '.join(f'{metric}: {getattr(self, metric):.3f}'
                        for metric, value in self.metrics.items()
                        if value is not None))


class Watcher(logging.getLoggerClass()):
    
    def __init__(self, log_path=None, rank=0):  # local-rank is the most important term
        super().__init__(name="watcher-transformer")
        
        self.rank = rank

        self.progress_bar = None
        self.best_tracker = None
        self.tb_writer    = None
        self.info_logger  = None

        if self.rank == 0:
        
            formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            if log_path is not None:
                fh = logging.FileHandler(log_path)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                self.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            self.addHandler(ch)
            self.setLevel(logging.DEBUG)

        else:
            self.setLevel(logging.CRITICAL)
    
    def info(self, msg, *args, **kwargs):
        if self.rank == 0:
            super().info(msg, *args, **kwargs)


    # ----- progress bar ---- #
    def close_progress_bar(self):
        if self.rank == 0:
            if self.progress_bar is not None:
                self.progressbar.close()

    def set_progress_bar(self, steps=0):
        if self.rank == 0:
            self.progressbar = tqdm(total=steps, desc="start a new progress-bar", position=0)

    def step_progress_bar(self, info_str=None, step=1):
        if self.rank == 0:
            self.progressbar.update(step)
            if info_str is not None:
                self.progressbar.set_description(info_str)

    # ----- tensorboard ---- #
    def set_tensorboard(self, path):
        if self.rank == 0:
            self.tb_writer = SummaryWriter(path)
    
    def add_tensorboard(self, name, value, iters, dtype='scalar'):
        if self.rank == 0:
            if dtype == 'scalar':
                self.tb_writer.add_scalar(name, value, iters)
            elif dtype == 'figure':
                self.tb_writer.add_figure(name, value, iters)
            else:
                raise NotImplementedError

    # ----- best performance tracker ---- #
    def set_best_tracker(self, model, opt, save_path, device, *names):
        self.best_tracker = Best(max, *names, 'i', model=model, opt=opt, path=save_path, gpu=device)

    def acc_best_tracker(self, iters, *values):
        if self.rank == 0:
            self.best_tracker.accumulate(*values, iters)



def masked_sort(x, mask, dim=-1):
    x.data += ((1 - mask) * INF).long()
    y, i = torch.sort(x, dim)
    y.data *= mask.long()
    return y, i

def unsorted(y, i, dim=-1):
    z = Variable(y.data.new(*y.size()))
    z.scatter_(dim, i, y)
    return z


def merge_cache(decoding_path, names0, last_epoch=0, max_cache=20):
    file_lock = open(decoding_path + '/_temp_decode', 'w')

    for name in names0:
        filenames = []
        for i in range(max_cache):
            filenames.append('{}/{}.ep{}'.format(decoding_path, name, last_epoch - i))
            if (last_epoch - i) <= 0:
                break
        code = 'cat {} > {}.train.{}'.format(" ".join(filenames), '{}/{}'.format(decoding_path, name), last_epoch)
        os.system(code)
    os.remove(decoding_path + '/_temp_decode')



#=====START: ADDED FOR DISTRIBUTED======

def gather_tensor(tensor, world_size=1):
    tensor_list = [tensor.clone() for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt / dist.get_world_size()

def reduce_dict(info_dict):
    for it in info_dict:
        p = info_dict[it].clone()
        dist.all_reduce(p, op=dist.reduce_op.SUM)
        info_dict[it] = p / dist.get_world_size()

def all_gather_list(data, max_size=32768):

    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size)
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 3 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 3))
    
    assert max_size < 255 * 255 * 256
    in_buffer[0] = enc_size // (255 * 255)  # this encoding works for max_size < 16M
    in_buffer[1] = (enc_size % (255 * 255)) // 255
    in_buffer[2] = (enc_size % (255 * 255)) % 255
    in_buffer[3:enc_size+3] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    result = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * 255 * item(out_buffer[0])) + 255 * item(out_buffer[1]) + item(out_buffer[2])
        result.append(
            pickle.loads(bytes(out_buffer[3:size+3].tolist()))
        )
    return result

def gather_dict(info_dict):
    for w in info_dict:
        new_v = []

        for v in all_gather_list(info_dict[w], 2 ** 19):
            if isinstance(v, list):
                new_v += v
            else:
                new_v.append(v)

        info_dict[w] = new_v


#=====END:   ADDED FOR DISTRIBUTED======