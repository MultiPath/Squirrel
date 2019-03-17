from contextlib import ExitStack

import numpy as np

from squirrel.data.field import Example


def lazy_reader_shuffled(paths, fields, buffer=16384, noise_generators=None):
    # -- infinite lazy dataloader --
    examples = []
    all_data = []
    out_step = 0
    epoch = 0

    while True:
        epoch += 1

        if epoch == 1:  # lazy reading the data

            with ExitStack() as stack:
                files = [
                    stack.enter_context(open(fname, "r", encoding="utf-8"))
                    for fname in paths
                ]
                for steps, lines in enumerate(zip(*files)):

                    lines = [line.strip() for line in lines]
                    if not any(line == '' for line in lines):
                        examples.append(lines)
                        all_data.append(lines)
                        out_step += 1

                    if (out_step % buffer == 0) and (out_step > 0):
                        # examples = sorted(examples, key=lambda x: sum([len(xi.split()) for xi in x]) )
                        for it, example in enumerate(examples):
                            yield Example.fromlist(example, fields,
                                                   it + out_step - buffer,
                                                   noise_generators)

                        examples = []
        else:

            # shuffle all data on the fly
            np.random.shuffle(all_data)
            for steps, lines in enumerate(all_data):

                examples.append(lines)
                out_step += 1

                if (out_step % buffer == 0) and (out_step > 0):
                    # examples = sorted(examples, key=lambda x: sum([len(xi.split()) for xi in x]) )
                    for it, example in enumerate(examples):
                        yield Example.fromlist(example, fields,
                                               it + out_step - buffer,
                                               noise_generators)

                    examples = []


def full_reader(paths, fields):
    with ExitStack() as stack:
        files = [
            stack.enter_context(open(fname, "r", encoding="utf-8"))
            for fname in paths
        ]
        examples = []
        for steps, lines in enumerate(zip(*files)):
            lines = [line.strip() for line in lines]
            if not any(line == '' for line in lines):
                examples.append(Example.fromlist(lines, fields, steps))

        return examples
