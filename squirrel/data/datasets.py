import copy
import os

from torchtext import data, datasets

from squirrel.data.batch import DistributedBatch, fetch_batch, fetch_pool
from squirrel.data.reader import full_reader, lazy_reader_shuffled

# BERT support
# from squirrel.bert_models.tokenization import whitespace_tokenize,
# BasicTokenizer, BertTokenizer
""" parallel dataset. using the lazy loader for training """


class ParallelDataset(datasets.TranslationDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input"""

    def __init__(self,
                 path=None,
                 exts=None,
                 fields=None,
                 lazy=True,
                 buffer=16384,
                 task=None,
                 noise_generators=None,
                 **kwargs):

        # assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)
        self.task = path.split('/')[-2] if task is None else task
        paths = tuple(os.path.expanduser(path + x) for x in exts)

        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        if lazy:  # using lazy dataloader -- cannot be used to construct vocab --
            new_fields = copy.deepcopy(fields)
            if noise_generators is not None:
                for i, (name, field) in enumerate(fields):
                    if noise_generators[i] is not None:
                        new_fields.append((name + '_n', field))

            super(datasets.TranslationDataset, self).__init__(
                lazy_reader_shuffled(
                    paths,
                    fields,
                    buffer=buffer,
                    noise_generators=noise_generators), new_fields, **kwargs)
        else:
            super(datasets.TranslationDataset, self).__init__(
                full_reader(paths, fields), fields, **kwargs)

    @classmethod
    def splits(cls,
               path,
               train=None,
               validation=None,
               test=None,
               lazy=True,
               **kwargs):
        train_data = None if train is None else cls(
            path + train, lazy=True, **kwargs)
        val_data = None if validation is None else cls(
            path + validation, lazy=False, **kwargs)
        test_data = None if test is None else cls(
            path + test, lazy=False, **kwargs)
        return train_data, val_data, test_data


""" A lazy verison of bucket iterator which supports saving unread minibatches. """


class LazyBucketIterator(data.BucketIterator):
    def __init__(self,
                 dataset,
                 batch_size,
                 sort_key=None,
                 device=None,
                 train=True,
                 repeat=None,
                 sort=None,
                 sort_within_batch=False,
                 distributed=False,
                 rank=0,
                 world_size=1,
                 maxlen=None,
                 maxatt_size=None,
                 init_tokens=None,
                 fields_for_batchsize=None):

        super().__init__(
            dataset,
            batch_size,
            sort_key,
            device,
            None,
            train,
            repeat,
            shuffle=False,
            sort=sort,
            sort_within_batch=sort_within_batch)

        # self.minibatch = []  # save unfinished batches.
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.maxatt_size = maxatt_size
        self.maxlen = maxlen
        self.message = None
        self.init_tokens = init_tokens
        self.task = dataset.task
        self.fields = dataset.fields.keys()
        self.fields_for_batchsize = fields_for_batchsize \
            if fields_for_batchsize is not None else self.fields

    def __iter__(self):
        count = 0
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

                if self.init_tokens is not None:
                    for name in self.init_tokens:
                        self.dataset.fields[
                            name].init_token = self.init_tokens[name]

                yield DistributedBatch(minibatch, self.dataset, self.device,
                                       self.world_size, self.rank, self.train)

            if not self.repeat:
                return

    def create_batches(self):
        if self.sort:
            self.batches = fetch_batch(
                self.data(),
                self.batch_size,
                self.world_size,
                True,
                maxlen=self.maxlen,
                maxatt_size=self.maxatt_size,
                fields=self.fields_for_batchsize)
        else:
            self.batches = fetch_pool(
                self.data(),
                self.batch_size,
                self.sort_key,
                random_shuffler=self.random_shuffler,
                world_size=self.world_size,
                maxlen=self.maxlen,
                maxatt_size=self.maxatt_size,
                fields=self.fields_for_batchsize)
