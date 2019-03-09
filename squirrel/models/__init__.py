import importlib
import os
import sys
import traceback
import hashlib
from pathlib import Path

FILE_ROOT = Path(__file__).parent
_MODELS = {}


def register_new_model(name):
    def register_new_model_fn(cl):
        if name in _MODELS:
            raise ValueError('Cannot register duplicated models')
        if not callable(cl):
            raise ValueError('Models must be callable ({name})')
        _MODELS[name] = cl
        return cl

    return register_new_model_fn


# automatically import any Python files in the scores/ directory
for m in os.listdir(FILE_ROOT):
    if m.endswith(('.py', '.pyc')) and not m.startswith('_'):
        model_name = m[:m.find('.py')]
        if model_name not in sys.modules:
            importlib.import_module('.' + model_name, 'squirrel.models')


def get_model(name):
    return _MODELS[name]