import logging
import operator
import os
import random
from collections import OrderedDict
from functools import reduce
from timeit import default_timer

import numpy as np
import pylab as plt
import seaborn as sns
import torch
from termcolor import colored
from tqdm import tqdm

from tensorboardX import SummaryWriter

# matplotlib.use('Agg')

params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (15, 5),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    'figure.max_open_warning': 200
}
plt.rcParams.update(params)
sns.set()


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prod(factors):
    return reduce(operator.mul, factors, 1)


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
    sns.heatmap(
        attention,
        ax=ax,
        cbar=False,
        cmap=sns.cubehelix_palette(start=2.4, rot=.1, light=1),
        square=True,
        xticklabels=seq1,
        yticklabels=seq2)
    ax.xaxis.tick_top()
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    return fig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        return values[0]  # loss

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key] / (self.count + 1e-9)
        raise AttributeError

    def __repr__(self):
        return (f"{self.name}: " + ', '.join(
            f'{metric}: {getattr(self, metric):.3f}'
            for metric, value in self.metrics.items() if value is not 0))

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
            if best_cmp_value is None or self.cmp_fn(best_cmp_value,
                                                     cmp_value) == cmp_value:

                self.metrics[cmp_metric] = cmp_value
                self.metrics.update({
                    metric: value
                    for metric, value in zip(
                        list(self.metrics.keys())[1:], other_values)
                })

                open(self.path + '.temp', 'w')
                if self.model is not None:
                    torch.save(self.model.state_dict(), self.path)
                if self.opt is not None:
                    if isinstance(self.opt, list):
                        torch.save(
                            [self.i, [opt.state_dict() for opt in self.opt]],
                            self.path + '.states')
                    else:
                        torch.save([self.i, self.opt.state_dict()],
                                   self.path + '.states')
                os.remove(self.path + '.temp')

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key]
        raise AttributeError

    def __repr__(self):
        return ("BEST: " + ', '.join(f'{metric}: {getattr(self, metric):.3f}'
                                     for metric, value in self.metrics.items()
                                     if value is not None))


class Watcher(logging.getLoggerClass()):
    def __init__(self, log_path=None,
                 rank=0):  # local-rank is the most important term
        super().__init__(name="watcher-transformer")

        self.rank = rank

        self.progress_bar = None
        self.best_tracker = None
        self.tb_writer = None
        self.info_logger = None

        if self.rank == 0:

            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s: - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

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
            self.progressbar = tqdm(
                total=steps, desc="start a new progress-bar", position=0)

    def step_progress_bar(self, info_str=None, step=1):
        if self.rank == 0:
            self.progressbar.update(step)
            if info_str is not None:
                self.progressbar.set_description(info_str)

    def set_languages(self, langs):
        try:
            from langid.langid import LanguageIdentifier, model
        except ImportError:
            print('Please install package of langid')

        self.langid = LanguageIdentifier.from_modelstring(
            model, norm_probs=True)
        try:
            self.langid.set_languages(langs)
        except ValueError:
            self.langid.set_languages(['en'])

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
            elif dtype == 'text':
                self.tb_writer.add_text(name, value, iters)
            else:
                raise NotImplementedError

    # ----- best performance tracker ---- #
    def set_best_tracker(self, model, opt, save_path, device, *names):
        self.best_tracker = Best(
            max, *names, 'i', model=model, opt=opt, path=save_path, gpu=device)

    def acc_best_tracker(self, iters, *values):
        if self.rank == 0:
            self.best_tracker.accumulate(*values, iters)

    def detect_lang(self, line):
        return self.langid.classify(line)[0]

    def match_lang(self, line, lang):
        scores = {l: v for l, v in self.langid.rank(line)}
        if lang not in scores:
            raise KeyError
        return scores[lang]


def masked_sort(x, mask, dim=-1):
    x.data += ((1 - mask) * INF).long()
    y, i = torch.sort(x, dim)
    y.data *= mask.long()
    return y, i


def unsorted(y, i, dim=-1):
    z = y.new(*y.size())
    z.scatter_(dim, i, y)
    return z


def merge_cache(decoding_path, names0, last_epoch=0, max_cache=20):
    for name in names0:
        filenames = []
        for i in range(max_cache):
            filenames.append('{}/{}.ep{}'.format(decoding_path, name,
                                                 last_epoch - i))
            if (last_epoch - i) <= 0:
                break
        code = 'cat {} > {}.train.{}'.format(
            " ".join(filenames), '{}/{}'.format(decoding_path, name),
            last_epoch)
        os.system(code)
    os.remove(decoding_path + '/_temp_decode')
