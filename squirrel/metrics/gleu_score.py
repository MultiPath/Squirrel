from nltk.translate.gleu_score import sentence_gleu, corpus_gleu

#################################################################################################
#                                                                                               #
#  Wrapper Function                                                                             #
#                                                                                               #
#################################################################################################
from . import register_corpus_metric


@register_corpus_metric('GLEU')
def get_gleu_score(targets, decodes):
    return corpus_gleu([[t] for t in targets], [o for o in decodes])

#################################################################################################