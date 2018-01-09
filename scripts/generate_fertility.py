import sys
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

maxv = 0
for line in sys.stdin:
    sources = [int(a.split('-')[0]) for a in line.split()]
    try:
        max_src = max(sources)
        outputs = [0 for _ in range(max_src+1)]
        for s in sources:
            outputs[s] += 1
        if max(outputs) > maxv:
            maxv = max(outputs)
        print(' '.join([str(s) for s in outputs]))
    except Exception:
        print('')

logger.info(maxv)
