"""
-- "Lazy dataloader" for Squirrel --
"""
import math
import torch
import torch.nn as nn
import numpy as np
import time
import os, sys
import torch.distributed as dist
import logging

from torchtext import data, datasets, vocab
from torchtext.data.batch import Batch
from contextlib import ExitStack
from collections import OrderedDict

# ====================== Helper Functions =========================================== #

""" Byte-level Transformation """
def str2byte(string):
    byte = string.encode('utf-8').hex()
    return [byte[k: k+2] for k in range(0, len(byte), 2)]

def byte2str(byte):
    try:
        output = bytes.fromhex(''.join(byte)).decode('utf-8')
    except Exception as e:
        output = ''
    return output


"""" A Lazy text-reader """
def lazy_reader(paths, fields, max_len=None, buffer=16384):  # -- infinite lazy dataloader --
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

                if (out_step % buffer == 0) and (out_step > 0):    # pre-reading the dataset, and cached...
                    # examples = sorted(examples, key=lambda x: sum([len(xi.split()) for xi in x]) )
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
def fetch_batch(data, batch_size, world_size=1, reserve=False, maxlen=10000, maxatt_size=None):
    """Yield elements from data in chunks of batch_size.
    :: minibatch: a reference of list which the remaining of batches will always be there for fetching next time.
    """

    # --- dynamic batching function -- # 
    def dynamic_batching(new, i, tokens, maxatt):
        tokens = tokens + max(len(new.src), len(new.trg))
        maxatt = maxatt / (i - 1) if i > 1 else 0
        maxatt = max(len(new.src) ** 2, len(new.trg)  ** 2,  maxatt) * i
        return tokens, maxatt

    if batch_size == 1:  # speed-test: one sentence per batch.
        batch_size_fn = lambda new, count, sofar, maxatt: count
    else:
        batch_size_fn = dynamic_batching

    if maxatt_size is None:
        maxatt_size = 1e10  # infinite

    size_so_far = 0
    maxatt_so_far = 0
    t0 = time.time()

    minibatch = []
    if reserve:
        reserved_minibatch = []

    for it, ex in enumerate(data):
        
        if max(len(ex.src), len(ex.trg)) > maxlen:
            continue

        if reserve and (it < world_size):
            reserved_minibatch.append(ex)
            continue

        else:
            minibatch.append(ex)
        
        size_so_far, maxatt_so_far = batch_size_fn(ex, len(minibatch), size_so_far, maxatt_so_far)
        
        def check(a, ax, b, bx):
            if ((a == ax) and (b <= bx)):
                return 0
            if ((b == bx) and (a <= ax)):
                return 0
            if ((a > ax) or (b > bx)):
                return 1
            return -1

        status = check(size_so_far, batch_size * world_size, np.ceil(maxatt_so_far / world_size), maxatt_size)
        
        if (status == 0) and (len(minibatch) > world_size):         # make sure there is no empty batches coming out during testing.
            # print(maxatt_so_far, np.ceil(maxatt_so_far / world_size))
            yield minibatch
            minibatch, size_so_far, maxatt_so_far = [], 0, 0
            
        elif (status == 1) and (len(minibatch) > (world_size + 1)): # make sure there is no empty batches coming out during testing.
            # print(maxatt_so_far, np.ceil(maxatt_so_far / world_size))
            yield minibatch[:-1]
            minibatch = minibatch[-1:]
            size_so_far, maxatt_so_far = batch_size_fn(ex, 1, 0, 0)

    if reserve:
        minibatch += reserved_minibatch  # make sure there is no empty batches coming out during testing.
    yield minibatch

    
""" pool of batch fetcher """
def fetch_pool(data, batch_size, key, random_shuffler=None, world_size=1, maxlen=10000, maxatt_size=None):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle

    for p in fetch_batch(data, batch_size * 100, maxatt_size=None):
        p_batch = fetch_batch(sorted(p, key=key), batch_size, world_size, True, maxlen=maxlen, maxatt_size=maxatt_size) 
        for b in random_shuffler(list(p_batch)):
            yield b

# ====================== Supportive Functions =========================================== #

""" sequence data field """
class Seuqence(data.Field):

    def __init__(self, reverse_tokenize, shuffle=0, dropout=0, replace=0, **kwargs):
        super().__init__(**kwargs)
        self.reverse_tokenizer = reverse_tokenize
        self.shuffle, self.dropout, self.replace = shuffle, dropout, replace

    def word_shuffle(self, x):
        if self.shuffle == 0:
            return x
        return [x[i] for i in (np.random.uniform(0, self.shuffle, size=(len(x))) + np.arange(len(x))).argsort()]

    def word_dropout(self, x):
        if self.dropout == 0:
            return x   
        return [xi for xi, di in zip(x, (np.random.rand(len(x)) >= self.dropout).tolist()) if di == 1]

    def word_blank(self, x, tok='<unk>'):
        if self.replace == 0:
            return x 
        return [xi if di == 1 else tok for xi, di in zip(x, (np.random.rand(len(x)) >= self.replace).tolist())]

    def add_noise(self, x, noise_level=None):
        if noise_level is None:
            return x

        if noise_level == 'n1':
            c = np.random.choice(3)
        elif noise_level == 'n2':
            c = np.random.choice(4)
        elif noise_level == 'n3':
            c = 4
        else:
            raise NotImplementedError

        if c == 0:
            return self.word_shuffle(x)
        elif c == 1:
            return self.word_dropout(x)
        elif c == 2:
            return self.word_blank(x, self.unk_token)
        elif c == 3:
            return x
        elif c == 4:      
            return self.word_blank(self.word_dropout(self.word_shuffle(x)), self.unk_token)
        else:
            raise NotImplementedError

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def reverse(self, batch, width=1, return_saved_time=False, reverse_token=True):
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
        
        def count(ex):
            n_step = 0
            n_pad  = 0
            n_word = 0

            filtered = []
            decision = []

            for e in ex:
                if e == self.init_token:
                    continue

                if e == self.pad_token:
                    n_pad += 1
                    if n_word > 0:
                        n_step += 1
                        n_word = 0

                else:
                    if n_word < (width - 1):
                        n_word += 1
                        
                    else:
                        n_word = 0
                        n_step += 1
                    
                    if n_word == 1:
                        decision.append(0)
                    else:
                        decision.append(1)

                    filtered.append(e)
            
            saved_time = (n_step + (n_word == 0)) / (1 + len(filtered))
            accuracy = len(filtered) / (len(ex) + 1e-9)
            return filtered, saved_time, accuracy, decision

        if return_saved_time:
            batch_filtered, saved_time, accuracy, decisions = [], [], [], []
            for ex in batch:
                b, s, a, d = count(ex)
                batch_filtered.append(b)
                saved_time.append(s)
                accuracy.append(a)
                decisions.append(d)

        else:
            batch_filtered = [list(filter(filter_special, ex)) for ex in batch]

        if not reverse_token:
            return batch_filtered

        output = [self.reverse_tokenizer(ex) for ex in batch_filtered]
        if return_saved_time:
            return output, saved_time, accuracy, decisions

        return output

    def reapply_noise(self, data, noise):
        batch = self.reverse(data, reverse_token=False)
        batch = [self.add_noise(ex, noise) for ex in batch]
        return self.process(batch, device=data.get_device())


""" parallel dataset. using the lazy loader for training """
class ParallelDataset(datasets.TranslationDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path=None, exts=None, fields=None, lazy=True, max_len=None, buffer=16384, task=None, **kwargs):

        assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)
        self.task = path.split('/')[-2] if task is None else task
        paths = tuple(os.path.expanduser(path + x) for x in exts)

        if lazy:  # using lazy dataloader -- cannot be used to construct the vocabulary -- 
            super(datasets.TranslationDataset, self).__init__(lazy_reader(paths, fields, max_len, buffer=buffer), fields, **kwargs)
        else:
            super(datasets.TranslationDataset, self).__init__(full_reader(paths, fields, max_len), fields, **kwargs)

    @classmethod
    def splits(cls, path, train=None, validation=None, test=None, lazy=True, **kwargs):
        train_data = None if train is None else cls(path + train, lazy=lazy, **kwargs)
        val_data = None if validation is None else cls(path + validation, lazy=False, **kwargs)
        test_data = None if test is None else cls(path + test, lazy=False, **kwargs)
        return train_data, val_data, test_data


class DistributedBatch(Batch):

    def __init__(self, data=None, dataset=None, device=None, world_size=1, local_rank=0):
        """Create a Batch from a list of examples."""
        
        if data is not None:
            big_batch_size = len(data)
            mini_batch_size = int(math.floor(big_batch_size / world_size))
            additional_size = int(big_batch_size -  mini_batch_size * world_size)

            start_pos = local_rank if additional_size > local_rank else additional_size
            start_pos = start_pos + local_rank * mini_batch_size
            end_pos = (local_rank + 1) if additional_size > (local_rank + 1) else additional_size
            end_pos = end_pos + (local_rank + 1) * mini_batch_size


            # start_pos = (additional_size + mini_batch_size) * local_rank
            # end_pos = (additional_size + mini_batch_size) * (local_rank + 1)
            data = data[start_pos: end_pos]
            
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            
            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))
                    

""" A lazy verison of bucket iterator which supports saving unread minibatches. """
class LazyBucketIterator(data.BucketIterator):

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                train=True, repeat=None, sort=None,
                sort_within_batch=False, distributed=False, rank=0, 
                world_size=1, maxlen=10000, maxatt_size=None):
        super().__init__(dataset, batch_size, sort_key, device, None, 
                        train, repeat, shuffle=False, sort=sort, sort_within_batch=sort_within_batch)
        
        # self.minibatch = []  # save unfinished batches.
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.maxlen = maxlen
        self.maxatt_size = maxatt_size

    def create_batches(self):
        if self.sort:
            self.batches = fetch_batch(self.data(), self.batch_size, self.world_size, True, maxlen=self.maxlen, maxatt_size=self.maxatt_size)
        else:
            self.batches = fetch_pool(self.data(), self.batch_size, self.sort_key, random_shuffler=self.random_shuffler, 
                                    world_size=self.world_size, maxlen=self.maxlen, maxatt_size=self.maxatt_size)

    # --- wrap the iterator --- 
    def __iter__(self):

        count = 0
        t0 = time.time()
        while True:
            
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                count += 1
                
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue

                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    minibatch.sort(key=self.sort_key, reverse=True)
                yield DistributedBatch(minibatch, self.dataset, self.device, self.world_size, self.rank)

            if not self.repeat:
                return


# ========================= DataLoader for Distributed Transformer ==================================== #

class MultiDataLoader(object):

    def __init__(self, args, logger=None, build_vocab=False, vocab_file=None):

        if logger is None:
            logger = logging.getLogger()

        # -- default setting -- #
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
        if args.remove_dec_eos:
            TRG = Field(batch_first=True, **common_kwargs)
        else:
            TRG = Field(init_token='<init>', eos_token='<eos>', **common_kwargs)

        if args.share_embeddings:
            SRC = TRG
        elif args.remove_enc_eos:
            SRC = Field(batch_first=True, **common_kwargs)
        else:
            SRC = Field(init_token='<init>', eos_token='<eos>', **common_kwargs)

        self.SRC, self.TRG = SRC, TRG

        # -- languages -- #
        # e.g. we assume the language markers are using "en,fr,zh"  
        srcs = args.src.split(',')
        trgs = args.trg.split(',')

        # --- read the vocabulary -- #
        if args.base != 'byte':
            reverse = False
            
            if args.multi:
                assert vocab_file is not None, "Multi-lingual Training requires to compute vocabulary manually first."
                assert os.path.exists(os.path.join(args.data_prefix, args.dataset, vocab_file)), 'need to pre-compute the vocab'

            if vocab_file is None:  # not recommanded. use the default vocabulary file.
                pair = srcs[0] + '-' + trgs[0]
                data_path = os.path.join(args.data_prefix, args.dataset, pair)
                if not os.path.exists(data_path):
                    pair = trgs[0] + '-' + srcs[0]
                    data_path = os.path.join(args.data_prefix, args.dataset, pair)
                    reverse = True
                    if not os.path.exists(data_path):
                        raise IOError   
                vocab_file = '{}/vocab.{}.{}.{}.pt'.format(pair, pair, 's' if args.share_embeddings else 'n', 'c' if args.base == 'char' else 'w')
            src_vocab, trg_vocab = torch.load(os.path.join(args.data_prefix, args.dataset, vocab_file))
            
            if not reverse:
                SRC.vocab = src_vocab
                TRG.vocab = trg_vocab
            else:
                SRC.vocab = trg_vocab
                TRG.vocab = src_vocab

        else:
            # Byte-level model always use the same vocabulary.
            SRC.build_vocab([["{0:x}".format(a)] for a in range(256)])
            TRG.build_vocab([["{0:x}".format(a)] for a in range(256)])
    
        args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

        # --- build batch-iterator for Translation tasks. ---
        self.train, self.dev, self.test = [], [], []

        def get_iterator(src, trg):

            # find the data #
            pair = src + '-' + trg
            data_path = os.path.join(args.data_prefix, args.dataset, pair)
            exts=('.src', '.trg')
            reverse = False

            if not os.path.exists(data_path):

                # translation in a reverse direction #
                pair = trg + '-' + src
                data_path = os.path.join(args.data_prefix, args.dataset, pair)
                exts=('.trg', '.src')
                reverse = True
                
                if not os.path.exists(data_path):
                    raise NotImplementedError   

            # --- setup dataset (no lazy mode when building the vocab) --- #
            train_data, dev_data, test_data = ParallelDataset.splits(
                path= data_path + '/', lazy=True,
                train=args.train_set, validation=args.dev_set, test=args.test_set, 
                exts=exts, fields=[('src', SRC), ('trg', TRG)],
                buffer=16384 * args.world_size, task='{}-{}'.format(src, trg))
            logger.info('setup the dataset.')


            if train_data is not None:
                train = LazyBucketIterator(train_data, 
                                        batch_size=args.batch_size, 
                                        device=args.device,
                                        sort_key=sort_key,
                                        train=True, 
                                        repeat=None if args.mode == 'train' else False,
                                        sort_within_batch=True, 
                                        distributed=args.distributed, 
                                        rank=args.local_rank, world_size=args.world_size,
                                        maxlen=args.maxlen, maxatt_size=args.maxatt_size)
                                                
            if dev_data is not None:
                dev = LazyBucketIterator(dev_data, 
                                        batch_size=args.batch_size, 
                                        device=args.device,
                                        sort_key=sort_key,
                                        train=False, 
                                        repeat=False, 
                                        sort_within_batch=True, 
                                        distributed=args.distributed, 
                                        rank=args.local_rank, world_size=args.world_size)

                
            if test_data is not None:   
                test = data.BucketIterator(test_data, batch_size=args.batch_size, device=args.device, train=False)

            logger.info("training set: {}-{} successfully loaded.".format(src, trg))
            return train, dev, test

        for src, trg in zip(srcs, trgs):
            train, dev, test = get_iterator(src, trg)
            
            self.train.append(train) 
            self.dev.append(dev)
            self.test.append(test) 

