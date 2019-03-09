import importlib
import os
import sys
from pathlib import Path

FILE_ROOT = Path(__file__).parent
_CORPUS_METRICS = {}


def register_corpus_metric(name):
    def register_corpus_metric_fn(fn):
        if name in _CORPUS_METRICS:
            raise ValueError('Cannot register duplicated scorers')
        if not callable(fn):
            raise ValueError('Metrics must be callable ({name})')
        _CORPUS_METRICS[name] = fn
        return fn

    return register_corpus_metric_fn


# automatically import any Python files in the scores/ directory
for m in os.listdir(FILE_ROOT):
    if m.endswith(('.py', '.pyc')) and not m.startswith('_'):
        model_name = m[:m.find('.py')]
        if model_name not in sys.modules:
            importlib.import_module('.' + model_name, 'squirrel.metrics')


def compute_scores(targets, decodes, names):
    results = {}
    for name in names:
        if name not in _CORPUS_METRICS:
            raise KeyError('Cannot find registered metrics')
        results[name] = _CORPUS_METRICS[name](targets, decodes)
    return results
