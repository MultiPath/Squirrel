import os, sys
import random
from contextlib import ExitStack

examples = []
with ExitStack() as stack:
    files = [stack.enter_context(open(fname, "r", encoding="utf-8")) for fname in sys.argv[1:]]
    for lines in zip(*files):
        examples.append(lines)

# shuffle
random.shuffle(examples)

def fix_name(fname):
    f, sufix = os.path.splitext(fname)
    return f + '.shuf' + sufix

with ExitStack() as stack:
    files = [stack.enter_context(open(fix_name(fname), "w", encoding="utf-8")) for fname in sys.argv[1:]]
    for lines in examples:
        for i, line in enumerate(lines):
            files[i].write(line)

print('shuffle done.')