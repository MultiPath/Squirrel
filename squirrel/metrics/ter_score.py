import os
import tempfile

#################################################################################################
#                                                                                               #
#  Wrapper Function                                                                             #
#                                                                                               #
#################################################################################################
from . import register_corpus_metric

TER_PATH = '/private/home/jgu/software/tercom.7.25.jar'


@register_corpus_metric('TER')
def get_ter_score(targets, decodes):
    return corpus_ter([" ".join(t) for t in targets],
                      [" ".join(o) for o in decodes])


#################################################################################################


def corpus_ter(ref, hyp):

    if not os.path.exists(TER_PATH):
        raise FileNotFoundError(
            "Please download meteor-1.5.jar and set the TER_PATH")

    fr, pathr = tempfile.mkstemp()
    fh, pathh = tempfile.mkstemp()

    try:
        with os.fdopen(fr, 'w') as tmp:
            tmp.write(
                '\n'.join([r + ' ({})'.format(i)
                           for i, r in enumerate(ref)]) + '\n')
        with os.fdopen(fh, 'w') as tmp:
            tmp.write(
                '\n'.join([h + ' ({})'.format(i)
                           for i, h in enumerate(hyp)]) + '\n')

        R = os.popen('java -jar {ter_path} -r {ref_file} -h {hyp_file}'.format(
            ter_path=TER_PATH, hyp_file=pathh, ref_file=pathr)).read()
        R = [l for l in R.split('\n') if 'Total TER' in l]
        R = float(R[0].split()[2])

    finally:
        os.remove(pathr)
        os.remove(pathh)
