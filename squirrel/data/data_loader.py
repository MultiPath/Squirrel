"""
-- "Lazy dataloader" for Squirrel --
"""
import copy
import logging
import os
from collections import Counter
from concurrent import futures

import torch
from torchtext import vocab

from squirrel.data.datasets import LazyBucketIterator, ParallelDataset
from squirrel.data.field import Features, Symbols
from squirrel.data.noise import get_noise_generator
from squirrel.data.tokenization import get_tokenizer

from . import register_dataloader


class AsynchronousDataLoderWrapper(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.pool = futures.ThreadPoolExecutor(2)
        self.last_batch = self.pool.submit(self.get_next)

    def __getattr__(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError:
            try:
                return getattr(self.dataloader, attr)
            except AttributeError:
                raise AttributeError('Error to find the attributes.')

    def get_next(self):
        return next(self.iterator)

    def __iter__(self):
        return self

    def __next__(self):
        this_batch = self.last_batch.result()
        self.last_batch = self.pool.submit(self.get_next)
        return this_batch


@register_dataloader('default')
class DataLoader(object):
    def __init__(self, args, logger=None, build_vocab=False, vocab_file=None):

        self.args = args
        if logger is None:
            logger = logging.getLogger()

        logger.info("""
            Use default-dataloader:
            Please not that vocabulary will be swapped if direction is reversed."""
                    )

        init, eos, pad, unk = '<init>', '<eos>', '<pad>', '<unk>'
        TRG = self.prepare_field(args.base, init, eos, pad, unk)

        if args.source_field == 'text':
            if args.share_embeddings:
                SRC = TRG
            else:
                SRC = self.prepare_field(args.base, init, eos, pad, unk)
        elif args.source_field == 'image_feature':
            assert not args.share_embeddings, 'image caption cannot share the same field.'
            SRC = Features()

        else:
            raise NotImplementedError

        # -- languages -- #
        src, trg = args.src, args.trg
        test_src, test_trg = args.test_src, args.test_trg
        assert (test_src == src) and (
            test_trg == trg), 'default only works for the same dataset.'

        # ------------------------ Vocabulary ----------------------------- #
        if self.args.base == 'byte':
            byte_vocab = [init, eos, pad, unk
                          ] + ["{0:x}".format(a) for a in range(256)]
            byte_vocab = self.prepare_vocabulary(token_list=byte_vocab)[0]
            SRC.vocab = byte_vocab
            TRG.vocab = byte_vocab

        else:

            assert vocab_file is not None, "compute vocabulary manually first."
            if not os.path.exists(vocab_file):
                vocab_file = os.path.join(args.data_prefix, args.dataset,
                                          vocab_file)
                if not os.path.exists(vocab_file):
                    raise FileNotFoundError(vocab_file)

                src_vocab, trg_vocab = self.prepare_vocabulary(
                    vocab_file=vocab_file)

                SRC.vocab = src_vocab
                TRG.vocab = trg_vocab

        # ----------------- de-couple the fields -------------- #
        SRC = copy.deepcopy(SRC)

        # ------------------------ DataPath ------------------------------ #
        data_path, reverse = self.find_data(src, trg)
        suffixes = self.prepare_suffix(self.args.suffixes, reverse)

        if reverse:
            _temp = copy.deepcopy(SRC.vocab)
            SRC.vocab = copy.deepcopy(TRG.vocab)
            TRG.vocab = _temp

        args.__dict__.update({
            'trg_vocab': len(TRG.vocab),
            'src_vocab': len(SRC.vocab)
        })
        self.SRC, self.TRG = SRC, TRG

        # --- build batch-iterator for Translation tasks. ---
        train, dev, test = self.get_data_iterator(
            data_path, suffixes, task='{}-{}'.format(src, trg), datasets='all')
        self.train, self.dev, self.test = [train], [dev], [test]

    def prepare_field(self,
                      base='bpe',
                      init_token='<init>',
                      eos_token='<eos>',
                      pad_token='<pad>',
                      unk_token='<unk>',
                      pretrained_vocab=None):

        tokenizer = get_tokenizer(base)(vocab=pretrained_vocab)

        common_kwargs = {
            'batch_first': True,
            'tokenize': tokenizer.tokenize,
            'reverse_tokenize': tokenizer.reverse,
            'pad_token': pad_token,
            'unk_token': unk_token,
            'init_token': init_token,
            'eos_token': eos_token
        }

        return Symbols(**common_kwargs)

    def prepare_vocabulary(self, vocab_file=None, token_list=None):
        if vocab_file is not None:
            if not os.path.exists(vocab_file):
                vocab_file = os.path.join(self.args.data_prefix,
                                          self.args.dataset, vocab_file)
                if not os.path.exists(vocab_file):
                    raise FileNotFoundError(vocab_file)
            vs = torch.load(vocab_file)

        elif token_list is not None:
            v = vocab.Vocab(counter=Counter())
            v.itos = token_list
            v.stoi = {d: i for i, d in enumerate(token_list)}
            vs = [v]

        else:
            raise NotImplementedError('We need a list or saved file.')

        return vs

    def find_data(self, src, trg):

        pair = src + '-' + trg
        data_path = os.path.join(self.args.data_prefix, self.args.dataset,
                                 pair)
        reverse = False

        if not os.path.exists(data_path):

            # translation in a reverse direction #
            pair = trg + '-' + src
            data_path = os.path.join(self.args.data_prefix, self.args.dataset,
                                     pair)
            reverse = True

            if not os.path.exists(data_path):
                raise FileNotFoundError

        return data_path, reverse

    def prepare_suffix(self, suffixes, reverse=False):
        if not reverse:
            return suffixes
        return [suffixes[1], suffixes[0]] + suffixes[2:]

    def prepare_init_tokens(self, src, trg):
        if not self.args.lang_as_init_token:
            init_tokens = None

        else:
            init_tokens = dict()
            translate_from = self.args.force_translate_from
            translate_to = self.args.force_translate_to

            init_tokens['src'] = '<{}>'.format(
                src if translate_from is None else translate_from)

            init_tokens['trg'] = '<{}>'.format(
                trg if translate_to is None else translate_to)

        return init_tokens

    def get_data_iterator(self,
                          data_path,
                          suffixes,
                          task='nmt',
                          datasets='train',
                          fields=None,
                          init_tokens=None,
                          sort_key=None,
                          fields_for_batchsize=None):

        if datasets == 'train':
            train_set, dev_set, test_set = self.args.train_set, None, None
        elif datasets == 'test':
            train_set, dev_set, test_set = None, self.args.dev_set, self.args.test_set
        elif datasets == 'all':
            train_set, dev_set, test_set = self.args.train_set, self.args.dev_set, self.args.test_set
        else:
            raise NotImplementedError('Unknown datasets')

        if fields is None:
            fields = [('src', self.SRC), ('trg', self.TRG)]
        noise_generators = [a[1].noise_generator for a in fields]

        # --- setup dataset (no lazy mode when building the vocab) --- #
        train_data, dev_data, test_data = ParallelDataset.splits(
            path=data_path + '/',
            lazy=True,
            train=train_set,
            validation=dev_set,
            test=test_set,
            exts=suffixes,
            fields=fields,
            buffer=16384 * self.args.world_size,
            task=task,
            noise_generators=noise_generators)

        train, dev, test = None, None, None
        if train_data is not None:
            train = AsynchronousDataLoderWrapper(
                LazyBucketIterator(
                    train_data,
                    batch_size=self.args.batch_size,
                    device=self.args.device,
                    sort_key=sort_key,
                    train=True,
                    repeat=None,
                    sort_within_batch=True,
                    distributed=self.args.distributed,
                    rank=self.args.local_rank,
                    world_size=self.args.world_size,
                    maxlen=self.args.maxlen,
                    maxatt_size=self.args.maxatt_size,
                    init_tokens=init_tokens,
                    fields_for_batchsize=fields_for_batchsize))

        if dev_data is not None:
            # dev = AsynchronousDataLoderWrapper(
            dev = LazyBucketIterator(
                dev_data,
                batch_size=self.args.valid_batch_size,
                device=self.args.device,
                sort_key=None,
                train=False,
                repeat=False,
                sort_within_batch=True,
                distributed=self.args.distributed,
                rank=self.args.local_rank,
                world_size=self.args.world_size,
                maxlen=self.args.valid_maxlen,
                init_tokens=init_tokens)

        if test_data is not None:
            # test = AsynchronousDataLoderWrapper(
            test = LazyBucketIterator(
                test_data,
                batch_size=self.args.valid_batch_size,
                device=self.args.device,
                sort_key=None,
                train=False,
                repeat=False,
                sort_within_batch=True,
                distributed=self.args.distributed,
                rank=self.args.local_rank,
                world_size=self.args.world_size,
                maxlen=self.args.valid_maxlen,
                init_tokens=init_tokens)

        return train, dev, test


@register_dataloader('multi')
class MultiDataLoader(DataLoader):
    """
    dataloader for general purpose
    """

    def __init__(self, args, logger=None, build_vocab=False, vocab_file=None):

        self.args = args

        if logger is None:
            logger = logging.getLogger()
        logger.info("""
            Use multi-dataloader:
            Please note that vocabulary (src, trg) is shared for all pairs""")

        init, eos, pad, unk = '<init>', '<eos>', '<pad>', '<unk>'
        TRG = self.prepare_field(args.base, init, eos, pad, unk)
        if args.share_embeddings:
            SRC = TRG
        else:
            SRC = self.prepare_field(args.base, init, eos, pad, unk)

        # -- languages -- #
        # e.g. we assume the language markers are using "en,fr,zh"
        srcs, trgs = args.srcs, args.trgs
        test_srcs, test_trgs = args.test_srcs, args.test_trgs

        # ------------------------ Vocabulary ----------------------------- #
        if self.args.base == 'byte':
            byte_vocab = [init, eos, pad, unk
                          ] + ["{0:x}".format(a) for a in range(256)]
            byte_vocab = self.prepare_vocabulary(token_list=byte_vocab)[0]
            SRC.vocab = byte_vocab
            TRG.vocab = byte_vocab

        else:

            assert vocab_file is not None, "compute vocabulary manually first."
            if not os.path.exists(vocab_file):
                vocab_file = os.path.join(args.data_prefix, args.dataset,
                                          vocab_file)
                if not os.path.exists(vocab_file):
                    raise FileNotFoundError(vocab_file)

                src_vocab, trg_vocab = self.prepare_vocabulary(
                    vocab_file=vocab_file)

                SRC.vocab = src_vocab
                TRG.vocab = trg_vocab

        # ----------------- Language ID as the Init-token ----- #

        if args.lang_as_init_token:
            additional_tokens = [
                '<{}>'.format(k) for k in sorted(args.all_langs)
            ]
            for token in additional_tokens:
                if token not in TRG.vocab.stoi:
                    TRG.vocab.stoi[token] = len(TRG.vocab.stoi)
                    TRG.vocab.itos.append(token)

                if not args.share_embeddings:
                    if token not in SRC.vocab.stoi:
                        SRC.vocab.stoi[token] = len(SRC.vocab.stoi)
                        SRC.vocab.itos.append(token)

        # ----------------- de-couple the fields -------------- #
        SRC = copy.deepcopy(SRC)

        args.__dict__.update({
            'trg_vocab': len(TRG.vocab),
            'src_vocab': len(SRC.vocab)
        })
        self.SRC, self.TRG = SRC, TRG

        # --- build batch-iterator for Translation tasks. ---
        self.train, self.dev, self.test = [], [], []

        for src, trg in zip(srcs, trgs):
            data_path, reverse = self.find_data(src, trg)
            suffixes = self.prepare_suffix(self.args.suffixes, reverse)
            train, _, _ = self.get_data_iterator(
                data_path,
                suffixes,
                task='{}-{}'.format(src, trg),
                datasets='train',
                init_tokens=self.prepare_init_tokens(src, trg))
            self.train.append(train)

        for src, trg in zip(test_srcs, test_trgs):
            data_path, reverse = self.find_data(src, trg)
            suffixes = self.prepare_suffix(self.args.suffixes, reverse)
            _, dev, test = self.get_data_iterator(
                data_path,
                suffixes,
                task='{}-{}'.format(src, trg),
                datasets='test',
                init_tokens=self.prepare_init_tokens(src, trg))
            self.dev.append(dev)
            self.test.append(test)


@register_dataloader('order')
class OrderDataLoader(DataLoader):
    """
    special dataloader only useful for Insertable Transformer (maybe not work).
    currently only supports BPE for simplicity.
    """

    def __init__(self, args, logger=None, build_vocab=False, vocab_file=None):

        self.args = args
        if logger is None:
            logger = logging.getLogger()

        logger.info("""
            Use ordered-dataloader:
            Please note that we need three files, .src, .trg and .pos""")

        init, eos, pad, unk = '<init>', '<eos>', '<pad>', '<unk>'
        TRG = self.prepare_field(args.base, init, eos, pad, unk)

        if args.source_field == 'text':
            if args.share_embeddings:
                SRC = TRG
            else:
                SRC = self.prepare_field(args.base, init, eos, pad, unk)

        elif args.source_field == 'image_feature':
            assert not args.share_embeddings, 'image caption cannot share the same field.'
            SRC = Features()

        else:
            raise NotImplementedError
        POS = self.prepare_field(args.base, None, None, pad, unk)

        # -- languages -- #
        src, trg = args.src, args.trg

        # --- read the vocabulary -- #
        assert vocab_file is not None, "compute vocabulary manually first"
        if not os.path.exists(vocab_file):
            vocab_file = os.path.join(args.data_prefix, args.dataset,
                                      vocab_file)
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(vocab_file)

            src_vocab, trg_vocab = self.prepare_vocabulary(
                vocab_file=vocab_file)

            SRC.vocab = src_vocab
            TRG.vocab = trg_vocab

        # ----------------- de-couple the fields -------------- #
        SRC = copy.deepcopy(SRC)
        POS.vocab = self.prepare_vocabulary(
            token_list=[pad, unk] + [str(i) for i in range(5001)])[0]

        # ------------------------ DataPath ------------------------------ #
        data_path, reverse = self.find_data(src, trg)
        if reverse:
            raise NotImplementedError(
                "Insertion mode cannot reverse the direction")
        suffixes = self.args.suffixes[:2] + ['.pos']

        args.__dict__.update({
            'trg_vocab': len(TRG.vocab),
            'src_vocab': len(SRC.vocab),
        })
        self.SRC, self.TRG, self.POS = SRC, TRG, POS

        # --- build batch-iterator for Translation tasks. ---
        train, _, _ = self.get_data_iterator(
            data_path,
            suffixes,
            task='{}-{}'.format(src, trg),
            datasets='train',
            fields=[('src', self.SRC), ('trg', self.TRG), ('pos', self.POS)])
        _, dev, test = self.get_data_iterator(
            data_path,
            suffixes[:2],
            task='{}-{}'.format(src, trg),
            datasets='test')
        self.train, self.dev, self.test = [train], [dev], [test]


@register_dataloader('target_noise')
class NoisyTargetDataLoder(DataLoader):
    def __init__(self, args, logger=None, build_vocab=False, vocab_file=None):

        self.args = args

        if not (self.args.noise_dataflow == 'trg'
                and self.args.noise_types is not None):
            raise NotImplementedError('we need to set the noise types')

        if logger is None:
            logger = logging.getLogger()

        logger.info("""
            Use noisy-dataloader (target side):
            Please not that vocabulary will be swapped if direction is reversed."""
                    )

        init, eos, pad, unk = '<init>', '<eos>', '<pad>', '<unk>'
        TRG = self.prepare_field(args.base, init, eos, pad, unk)

        if args.source_field == 'text':
            if args.share_embeddings:
                SRC = TRG
            else:
                SRC = self.prepare_field(args.base, init, eos, pad, unk)
        elif args.source_field == 'image_feature':
            assert not args.share_embeddings, 'image caption cannot share the same field.'
            SRC = Features()

        else:
            raise NotImplementedError

        # -- languages -- #
        src, trg = args.src, args.trg
        test_src, test_trg = args.test_src, args.test_trg
        assert (test_src == src) and (
            test_trg == trg), 'default only works for the same dataset.'

        # ------------------------ Vocabulary ----------------------------- #
        if self.args.base == 'byte':
            byte_vocab = [init, eos, pad, unk
                          ] + ["{0:x}".format(a) for a in range(256)]
            byte_vocab = self.prepare_vocabulary(token_list=byte_vocab)[0]
            SRC.vocab = byte_vocab
            TRG.vocab = byte_vocab

        else:

            assert vocab_file is not None, "compute vocabulary manually first."
            if not os.path.exists(vocab_file):
                vocab_file = os.path.join(args.data_prefix, args.dataset,
                                          vocab_file)
                if not os.path.exists(vocab_file):
                    raise FileNotFoundError(vocab_file)

                src_vocab, trg_vocab = self.prepare_vocabulary(
                    vocab_file=vocab_file)

                SRC.vocab = src_vocab
                TRG.vocab = trg_vocab

        # ----------------- de-couple the fields -------------- #
        SRC = copy.deepcopy(SRC)

        # ------------------------ DataPath ------------------------------ #
        data_path, reverse = self.find_data(src, trg)
        suffixes = self.prepare_suffix(self.args.suffixes, reverse)

        if reverse:
            _temp = copy.deepcopy(SRC.vocab)
            SRC.vocab = copy.deepcopy(TRG.vocab)
            TRG.vocab = _temp

        # ----------------- Noisy Dataloader (target side) ---- #
        TRG.set_noise_generator(
            get_noise_generator(
                self.args.noise_types, {
                    a[6:]: getattr(self.args, a)
                    for a in self.args.__dict__ if a[:5] == 'noise'
                }, self.args.output_suggested_edits))

        args.__dict__.update({
            'trg_vocab': len(TRG.vocab),
            'src_vocab': len(SRC.vocab)
        })

        self.SRC, self.TRG = SRC, TRG

        # --- build batch-iterator for Translation tasks. ---
        train, dev, test = self.get_data_iterator(
            data_path,
            suffixes,
            task='{}-{}'.format(src, trg),
            datasets='all',
            sort_key=lambda x: len(x.trg_n))
        self.train, self.dev, self.test = [train], [dev], [test]
