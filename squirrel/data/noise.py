"""
This file describes different noisy functions
"""
import numpy as np

_NOISE_GENERATORS = {}


def register_noise_generator(name):
    def register_noise_generator_fn(fn):
        if name in _NOISE_GENERATORS:
            raise ValueError('Cannot register duplicated generators')
        if not callable(fn):
            raise ValueError('Generator must be callable ({name})')
        _NOISE_GENERATORS[name] = fn
        return fn

    return register_noise_generator_fn


class merged_noisy_generator(object):
    def __init__(self, generators, output_suggested_edits=False):
        self.generators = generators
        self.output_suggested_edits = output_suggested_edits

    def __call__(self, x):
        for g in self.generators:
            x = g(x)
        return x


def get_noise_generator(names, conditions, output_suggested_edits=False):
    return merged_noisy_generator(
        [_NOISE_GENERATORS[name](**conditions) for name in names],
        output_suggested_edits)


@register_noise_generator('word_shuffle')
class word_shuffle(object):
    def __init__(self, shuffle_distance=3, **kwargs):
        self.shuffle_distance = shuffle_distance

    def __call__(self, x):
        if self.shuffle_distance == 0:
            return x
        return [
            x[i] for i in (
                np.random.uniform(0, self.shuffle_distance, size=(len(x))) +
                np.arange(len(x))).argsort()
        ]


@register_noise_generator('word_dropout')
class word_dropout(object):
    def __init__(self, dropout_prob=0.0, **kwargs):
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        if self.dropout_prob == 0:
            return x
        return [
            xi for xi, di in zip(x, (
                np.random.rand(len(x)) >= self.dropout_prob).tolist())
            if di == 1
        ]


@register_noise_generator('word_blank')
class word_blank(object):
    def __init__(self, blank_prob=0.0, blank_word='<unk>', **kwargs):
        self.blank_prob = blank_prob
        self.blank_word = blank_word

    def __call__(self, x):
        if self.blank_prob == 0:
            return x
        return [
            xi if di == 1 else self.blank_word
            for xi, di in zip(x, (
                np.random.rand(len(x)) >= self.blank_prob).tolist())
        ]


@register_noise_generator('word_dropout_at_anywhere')
class word_dropout_at_anywhere(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, x):
        return [
            x[i] for i in np.sort(
                np.random.permutation(len(x))
                [:np.random.randint(0,
                                    len(x) + 1)])
        ]
