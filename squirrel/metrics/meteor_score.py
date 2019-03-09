import os, tempfile

METEOR_PATH = '/private/home/jgu/software/meteor-1.5/meteor-1.5.jar'

#################################################################################################
#                                                                                               #
#  Wrapper Function                                                                             #
#                                                                                               #
#################################################################################################
from . import register_corpus_metric


@register_corpus_metric('METEOR')
def get_meteor_score(targets, decodes):
    return corpus_meteor([" ".join(t) for t in targets], [" ".join(o) for o in decodes])

#################################################################################################

def corpus_meteor(ref, hyp, lang='other'):

    if not os.path.exists(METEOR_PATH):
        raise FileNotFoundError("Please download meteor-1.5.jar and set the METEOR_PATH")

    fr, pathr = tempfile.mkstemp()
    fh, pathh = tempfile.mkstemp()

    try:
        with os.fdopen(fr, 'w') as tmp:
            tmp.write('\n'.join([r + ' ({})'.format(i) for i, r in enumerate(ref)]) + '\n')
        with os.fdopen(fh, 'w') as tmp:
            tmp.write('\n'.join([h + ' ({})'.format(i) for i, h in enumerate(hyp)]) + '\n')

        R = os.popen('java -Xmx2G  -jar {jar_path} {hyp_file} {ref_file} -l {lang}'.format(
                        jar_path = METEOR_PATH, hyp_file=pathh, ref_file=pathr, lang=lang)).read()
        
        R = float([l for l in R.split('\n') if 'Final score' in l][0].split()[-1])
        #R = R[0].split()[2]

    finally:   
        os.remove(pathr)
        os.remove(pathh)

    return R