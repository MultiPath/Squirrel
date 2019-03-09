"""
-- Refactor the dataloader for Squirrel
"""

import math
import random

import numpy as np
import torch
from torchtext.data.batch import Batch


class DistributedBatch(Batch):
    def __init__(self,
                 data=None,
                 dataset=None,
                 device=None,
                 world_size=1,
                 local_rank=0):
        """Create a Batch from a list of examples."""

        self.message = ''
        self.task = ''
        self.preprocessed = None
        self.weights = None

        if data is not None:
            big_batch_size = len(data)
            mini_batch_size = int(math.floor(big_batch_size / world_size))
            additional_size = int(big_batch_size -
                                  mini_batch_size * world_size)

            start_pos = local_rank if additional_size > local_rank \
                else additional_size
            start_pos = start_pos + local_rank * mini_batch_size
            end_pos = (local_rank + 1) if additional_size > (
                local_rank + 1) else additional_size
            end_pos = end_pos + (local_rank + 1) * mini_batch_size

            data = data[start_pos:end_pos]

            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.task = dataset.task
            self.attributes = []

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]

                    setattr(self, name + '_original', batch)
                    setattr(self, name, field.process(batch, device=device))

                    self.attributes += [name, name + '_original']

            if hasattr(data[0], 'id'):
                setattr(
                    self, 'id',
                    torch.tensor([getattr(x, 'id') for x in data],
                                 dtype=torch.long,
                                 device=device))

                self.attributes += ['id']


def fetch_batch(data,
                batch_size,
                world_size=1,
                reserve=False,
                maxlen=None,
                maxatt_size=None,
                fields=['src', 'trg']):

    # --- dynamic batching function (by default) --- #
    def dynamic_batching(new, i, tokens, maxatt):
        tokens = tokens + max([len(getattr(new, field)) for field in fields])
        maxatt = maxatt / (i - 1) if i > 1 else 0
        maxatt = max([len(getattr(new, field))**2
                      for field in fields] + [maxatt]) * i
        return tokens, maxatt

    if batch_size == 1:  # speed-test: one sentence per batch.
        batch_size_fn = lambda new, count, sofar, maxatt: count
    else:
        batch_size_fn = dynamic_batching

    if maxatt_size is None:
        maxatt_size = 1e10  # infinite

    size_so_far = 0
    maxatt_so_far = 0

    minibatch = []
    if reserve:
        reserved_minibatch = []

    for it, ex in enumerate(data):

        # drop examples that has elements too long
        if maxlen is not None:
            if max([len(getattr(ex, field)) for field in fields]) > maxlen:
                continue

        if reserve and (it < world_size):
            reserved_minibatch.append(ex)
            continue

        else:
            minibatch.append(ex)

        size_so_far, maxatt_so_far = batch_size_fn(ex, len(minibatch),
                                                   size_so_far, maxatt_so_far)

        def check(a, ax, b, bx):
            if ((a == ax) and (b <= bx)):
                return 0
            if ((b == bx) and (a <= ax)):
                return 0
            if ((a > ax) or (b > bx)):
                return 1
            return -1

        status = check(size_so_far, batch_size * world_size,
                       np.ceil(maxatt_so_far / world_size), maxatt_size)

        if (status == 0) and (
                len(minibatch) > world_size
        ):  # make sure there is no empty batches coming out during testing.
            # print(maxatt_so_far, np.ceil(maxatt_so_far / world_size))
            yield minibatch
            minibatch, size_so_far, maxatt_so_far = [], 0, 0

        elif (status == 1) and (
                len(minibatch) > (world_size + 1)
        ):  # make sure there is no empty batches coming out during testing.
            # print(maxatt_so_far, np.ceil(maxatt_so_far / world_size))
            yield minibatch[:-1]
            minibatch = minibatch[-1:]
            size_so_far, maxatt_so_far = batch_size_fn(ex, 1, 0, 0)

    if reserve:
        minibatch += reserved_minibatch  # make sure there is no empty batches
    yield minibatch


def fetch_pool(data,
               batch_size,
               key,
               random_shuffler=None,
               world_size=1,
               maxlen=None,
               maxatt_size=None,
               fields=None):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle

    for p in fetch_batch(
            data, batch_size * 100, maxlen=maxlen, maxatt_size=None,
            fields=fields):  # pre-read 100 batches and sort.
        p_batch = fetch_batch(
            sorted(p, key=key),
            batch_size,
            world_size,
            True,
            maxlen=maxlen,
            maxatt_size=maxatt_size,
            fields=fields)
        for b in random_shuffler(list(p_batch)):
            yield b


def split_batch(batch, N):  # split a batch into N parts.
    if N == 1:
        yield batch

    else:
        backup_batch = batch
        big_batch_size = backup_batch.batch_size
        mini_batch_size = int(math.floor(big_batch_size / N))
        additional_size = int(big_batch_size - mini_batch_size * N)

        batches = []
        for k in range(N):
            batch = DistributedBatch()
            batch.fields = backup_batch.fields

            start_pos = k if additional_size > k else additional_size
            start_pos = start_pos + k * mini_batch_size
            end_pos = (k + 1) if additional_size > (k + 1) else additional_size
            end_pos = end_pos + (k + 1) * mini_batch_size

            if start_pos >= end_pos:
                continue

            batch.batch_size = end_pos - start_pos
            if backup_batch.preprocessed is not None:
                batch.preprocessed = []
                for u in range(len(backup_batch.preprocessed)):
                    batch.preprocessed.append(
                        backup_batch.preprocessed[u][start_pos:end_pos])

            if backup_batch.weights is not None:
                batch.weights = backup_batch.weights[start_pos:end_pos]

            for field in backup_batch.fields:
                setattr(batch, field,
                        getattr(backup_batch, field)[start_pos:end_pos])
                if hasattr(backup_batch, field + '_original'):
                    setattr(
                        batch, field + '_original',
                        getattr(backup_batch,
                                field + '_original')[start_pos:end_pos])

            if hasattr(backup_batch, 'id'):
                setattr(batch, 'id',
                        getattr(backup_batch, 'id')[start_pos:end_pos])

            batches.append(batch)

        for batch in batches:
            yield batch


def merge_batches(batches):  # merge batches into a big batch
    if len(batches) == 1:
        return batches[0]

    else:
        batch = DistributedBatch()

        for field in batches[-1].fields:
            for backup_batch in batches:
                assert field in backup_batch.fields, "the same fields"

        batch.fields = batches[-1].fields
        for field in batches[-1].fields:
            max_len = max([
                getattr(backup_batch, field).size(1)
                for backup_batch in batches
            ])
            setattr(
                batch, field,
                torch.cat([
                    backup_batch.dataset.fields[field].extend_padding(
                        getattr(backup_batch, field), max_len)
                    for backup_batch in batches
                ], 0))

            if hasattr(batches[-1], field + '_original'):
                setattr(batch, field + '_original', [
                    i for t in getattr(backup_batch, field + '_original')
                    for i in t
                ])

        if hasattr(batches[-1], 'id'):
            setattr(batch, 'id',
                    torch.cat([backup_batch.id for backup_batch in batches]))

        batch.batch_size = sum(
            [backup_batch.batch_size for backup_batch in batches])
        batch.task = '/'.join([backup_batch.task for backup_batch in batches])
        return batch
