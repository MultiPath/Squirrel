try:
    import fast_editdistance as ed
except ImportError:
    import editdistance as ed

import numpy as np

from . import register_corpus_metric


@register_corpus_metric('EDIT')
def get_ter_score(targets, decodes):
    targets = [[hash(ti) for ti in t] for t in targets]
    decodes = [[hash(oi) for oi in o] for o in decodes]

    return np.mean(ed.compute_ed2(decodes, targets))
