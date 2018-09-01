"""
-- "Lazy dataloader" for Squirrel --
"""

import torch
import torch.nn as nn
import numpy as np
import os
import torch.distributed as dist

from torchtext import data, datasets
from torchtext.data.batch import Batch
from contextlib import ExitStack
from collections import OrderedDict

# ====================== Supportive Functions =========================================== #

"""" A Lazy text-reader """
def lazy_reader(paths, fields, max_len=None, chache=16384):  # -- infinite lazy dataloader --
    examples = []
    out_step = 0

    while True:
        
        with ExitStack() as stack:
            files = [stack.enter_context(open(fname, "r", encoding="utf-8")) for fname in paths]         
            for steps, lines in enumerate(zip(*files)):
                
                lines = [line.strip() for line in lines]
                if not any(line == '' for line in lines):
                    if max_len is not None:
                        flag = 0
                        for line in lines:
                            if len(line.split()) > max_len:
                                flag = 1
                                break
                        if flag == 1:
                            continue   

                    examples.append(lines)
                    out_step += 1

                if (out_step % chache == 0) and (out_step > 0):    # pre-reading the dataset, and cached...
                    for it, example in enumerate(examples):
                        yield data.Example.fromlist(example, fields)

                    examples = []

"""" A Full text-reader """
def full_reader(paths, fields, max_len=None):
    with ExitStack() as stack:
        files = [stack.enter_context(open(fname, "r", encoding="utf-8")) for fname in paths]
        examples = []
        for steps, lines in enumerate(zip(*files)):
            lines = [line.strip() for line in lines]
            if not any(line == '' for line in lines):
                examples.append(data.Example.fromlist(lines, fields))
        return examples

""" batch fetcher """
def fetch_batch(minibatch, data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size.
    :: minibatch: a reference of list which the remaining of batches will always be there for fetching next time.
    """

    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count

    size_so_far = 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0

        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    
""" pool of batch fetcher """
def fetch_pool(minibatch, data, batch_size, key, batch_size_fn=lambda new, count, sofar: count, random_shuffler=None):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle

    for p in fetch_batch(minibatch, data, batch_size * 100, batch_size_fn):
        microbatch = []

        p_batch = fetch_batch(microbatch, sorted(p, key=key), batch_size, batch_size_fn)
        for b in random_shuffler(list(p_batch)):
            yield b

        minibatch += microbatch # collect remaining mini-batches.



# ====================== Supportive Functions =========================================== #

""" sequence data field """
class Seuqence(data.Field):

    def reverse(self, batch, char=False):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past frst eos
        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if not char:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]
        else:
            batch = ["".join(filter(filter_special, ex)) for ex in batch]
        return batch


""" parallel dataset. using the lazy loader for training """
class ParallelDataset(datasets.TranslationDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path=None, exts=None, fields=None, lazy=True, max_len=None, **kwargs):

        assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)
        paths = tuple(os.path.expanduser(path + x) for x in exts)

        if lazy:  # using lazy dataloader -- cannot be used to construct the vocabulary -- 
            super(datasets.TranslationDataset, self).__init__(lazy_reader(paths, fields, max_len), fields, **kwargs)
        else:
            super(datasets.TranslationDataset, self).__init__(full_reader(paths, fields, max_len), fields, **kwargs)

    @classmethod
    def splits(cls, path, train=None, validation=None, test=None, **kwargs):
        train_data = None if train is None else cls(path + train, lazy=True, **kwargs)
        val_data = None if validation is None else cls(path + validation, lazy=False, **kwargs)
        test_data = None if test is None else cls(path + test, lazy=False, **kwargs)
        return train_data, val_data, test_data


""" A lazy verison of bucket iterator which supports saving unread minibatches. """
class LazyBucketIterator(data.BucketIterator):

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True, repeat=None, 
                 sort_within_batch=False, distributed=False, rank=0, world_size=1):
        super().__init__(dataset, batch_size, sort_key, device, batch_size_fn, 
                         train, repeat, shuffle=False, sort=False, sort_within_batch=sort_within_batch)
        
        self.minibatch = []  # save unfinished batches.
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size

    def create_batches(self):
        self.batches = fetch_pool(self.minibatch, self.data(), self.batch_size,
                                    self.sort_key, self.batch_size_fn,
                                    random_shuffler=self.random_shuffler)

    # --- wrap the iterator --- 
    def __iter__(self):
        count = 0
        while True:
            
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                count += 1
                
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue

                # --- distributed iterator ---
                if self.distributed:
                    if count % self.world_size != self.rank:
                        continue

                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    minibatch.sort(key=self.sort_key, reverse=True)

                yield Batch(minibatch, self.dataset, self.device)

            if not self.repeat:
                return


# ========================= DataLoader for Distributed Transformer ==================================== #
class DataLoader(object):

    def __init__(self, args, logger):

        # --- setup data field --- #
        if not args.char:
            tokenizer = lambda s: s.split() 
        else:
            tokenizer = lambda s: list(s)
            
        if args.remove_dec_eos:
            TRG = Seuqence(batch_first=True, tokenize=tokenizer)
        else:
            TRG = Seuqence(init_token='<init>', eos_token='<eos>', batch_first=True, tokenize=tokenizer)

        if args.share_embeddings:
            SRC = TRG
        elif args.remove_enc_eos:
            SRC = Seuqence(batch_first=True, tokenize=tokenizer)
        else:
            SRC = Seuqence(init_token='<init>', eos_token='<eos>', batch_first=True, tokenize=tokenizer)

        self.SRC, self.TRG = SRC, TRG


        data_path = os.path.join(args.data_prefix, args.dataset, args.src + '-' + args.trg)

        # --- setup dataset --- #
        train_data, dev_data, test_data = ParallelDataset.splits(
            path= data_path + '/', 
            train=args.train_set, validation=args.dev_set, test=args.test_set, 
            exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)])

        # --- read the vocabulary -- #
        vocab_name = 'vocab.{}-{}.{}.{}.pt'.format(args.src, args.trg, 
                                                's' if args.share_embeddings else 'n',
                                                'c' if args.char else 'w')

        logger.info('load saved vocabulary.')
        assert os.path.exists(os.path.join(data_path, vocab_name)), 'need to pre-compute the vocab'
        src_vocab, trg_vocab = torch.load(os.path.join(data_path, vocab_name))

        SRC.vocab = src_vocab
        TRG.vocab = trg_vocab

        args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

        # --- dynamic batching function -- #
        def dyn_batch_with_padding(new, i, sofar):
            prev_max_len = sofar / (i - 1) if i > 1 else 0
            t =  max(len(new.src), len(new.trg),  prev_max_len) * i
            return t

        def dyn_batch_without_padding(new, i, sofar):
            return sofar + max(len(new.src), len(new.trg))

        if args.batch_size == 1:  # speed-test: one sentence per batch.
            batch_size_fn = lambda new, count, sofar: count

        else:
            batch_size_fn = dyn_batch_with_padding

        
        # --- build batch-iterator for Translation tasks. ---
        self.train, self.dev, self.test = None, None, None
        if train_data is not None:
            logger.info("build the training set.")
            self.train = LazyBucketIterator(train_data, 
                                            batch_size=args.batch_size, 
                                            device=args.device,
                                            batch_size_fn=batch_size_fn, train=True, 
                                            repeat=None if args.mode == 'train' else False,
                                            sort_within_batch=True, 
                                            distributed=args.distributed, 
                                            rank=args.local_rank, world_size=args.world_size)
        if dev_data is not None:
            logger.info("build the validation set. (normal iterator is fine)")
            self.dev = data.BucketIterator(dev_data, batch_size=args.batch_size * 4, device=args.device,
                                            batch_size_fn=batch_size_fn, train=False)
            
        if test_data is not None: 
            logger.info("build the testing set. (normal iterator is fine)")   
            self.test = data.BucketIterator(test_data, batch_size=args.batch_size * 8, device=args.device,
                                            batch_size_fn=batch_size_fn, train=False)


    