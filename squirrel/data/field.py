from multiprocessing import Pool

import numpy as np
import torch
from torchtext import data


"""
Example (a colleciton of text is one)
"""


class Example(data.Example):
    @classmethod
    def fromlist(cls, data, fields, step=None):
        ex = super().fromlist(data, fields)
        if step is not None:
            setattr(ex, 'id', step)
        return ex


"""
Text Field
"""


class Symbols(data.Field):
    def __init__(self, reverse_tokenize, additional_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self.reverse_tokenizer = reverse_tokenize
        self.additional_tokens = additional_tokens if additional_tokens is not None else []
        self.name = 'symbols'

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def extend_padding(self, batch, maxlen):
        new_batch = batch.new_zeros(batch.size(0), maxlen).fill_(
            self.vocab.stoi[self.pad_token])
        new_batch[:, :batch.size(1)] = batch
        return new_batch

    def reverse(self,
                batch,
                width=1,
                return_saved_time=False,
                reverse_token=False):

        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex]
                 for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token)
                 for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch_filtered = [list(filter(filter_special, ex)) for ex in batch]

        if not reverse_token:
            return batch_filtered

        output = [self.reverse_tokenizer(ex) for ex in batch_filtered]
        return output


""" 
COCO image features field (only useful for image caption experiments) 
"""


class Features(data.Field):
    def __init__(self, map_size=7, feature_size=512, workers=8, **kwargs):
        super().__init__(**kwargs)
        self.map_size = map_size
        self.feature_size = feature_size
        self.name = 'features'
        self.reverse_tokenizer = lambda x: x[0]
        self.tokenizer = lambda x: [x]
        self.data_dir = None

    def set_datapath(self, data_dir):
        self.data_dir = data_dir

    def process(self, batch, device=None):
        if self.data_dir is None:
            raise FileNotFoundError('Must set an image path first')

        with Pool(8) as pool:
            arr = np.array(
                pool.map_async(np.load,
                               [self.data_dir + '/' + x[0]
                                for x in batch]).get())
            tensor = torch.from_numpy(arr).to(device).view(
                -1, self.feature_size,
                self.map_size * self.map_size).transpose(1, 2).contiguous()
        return tensor

    def reverse(self, batch):
        return ['image features' for _ in range(batch.size(0))]
