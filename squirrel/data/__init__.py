import importlib

_DATALOADERS = {}


def register_dataloader(name):
    def register_new_dataloader_fn(fn):
        if name in _DATALOADERS:
            raise ValueError('Cannot register duplicated scorers')
        if not callable(fn):
            raise ValueError('Dataloader must be callable ({name})')
        _DATALOADERS[name] = fn
        return fn

    return register_new_dataloader_fn


importlib.import_module('squirrel.data.data_loader')


def get_dataloader(name):
    if name not in _DATALOADERS:
        raise KeyError('Cannot find registered dataloader')
    return _DATALOADERS[name]
