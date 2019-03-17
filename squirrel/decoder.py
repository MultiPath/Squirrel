import os
import time
from collections import defaultdict

import numpy as np
import torch

from squirrel.data.tokenization import get_tokenizer
from squirrel.metrics import compute_scores
from squirrel.utils import gather_dict


def valid_model_ppl(args,
                    watcher,
                    model,
                    dev,
                    dataflow=['src', 'trg'],
                    task=None):
    model.eval()
    outputs = defaultdict(lambda: [])
    watcher.set_progress_bar(len(dev.dataset))

    mode = 'valid'
    if task is not None:
        mode = mode + ',' + task

    for j, dev_batch in enumerate(dev):

        dev_outputs = model(dev_batch, mode=mode, dataflow=dataflow)

        if hasattr(dev_batch, 'id'):
            dev_outputs['id'] = dev_batch.id

        for key in dev_outputs:
            if isinstance(dev_outputs[key], list):
                continue
            try:
                dev_outputs[key] = [dev_outputs[key].item()]
            except ValueError:
                dev_outputs[key] = dev_outputs[key].cpu().tolist()

        if args.distributed:
            gather_dict(dev_outputs)

        for key in dev_outputs:
            if isinstance(dev_outputs[key], list):
                outputs[key] += dev_outputs[key]
            else:
                outputs[key] += [dev_outputs[key]]

        info_str = 'Valid: sentences={}, ppl={:.3f}'.format(
            sum(outputs['sents']), np.mean(outputs['loss']))
        watcher.step_progress_bar(
            info_str=info_str, step=sum(dev_outputs['sents']))
    watcher.close_progress_bar()

    # record for tensorboard:
    outputs['tb_data'] += [('dev/{}/PPL'.format(dev.dataset.task),
                            np.mean(outputs['loss']))]
    return outputs


def valid_model(args,
                watcher,
                model,
                dev,
                dataflow=['src', 'trg'],
                print_out=False,
                decoding_path=None):

    model.eval()

    curr_time = 0
    outputs = defaultdict(lambda: [])
    watcher.set_progress_bar(len(dev.dataset))

    tokenizer, space_tokenizer = get_tokenizer(
        args.base)(), get_tokenizer('space')()

    tokenizer, space_tokenizer = get_tokenizer(
        args.base)(), get_tokenizer('space')()

    def eval_pipe(x):
        return space_tokenizer.tokenize(tokenizer.reverse(x))

    dev.noise_flow = dataflow[0]
    fh, output_handles = None, None

    # output the sequences
    if (decoding_path is not None) and (args.local_rank == 0):

        output_flows = ['src', 'trg', 'dec']
        if args.insertable:
            output_flows += ['ori']

        if args.output_decoding_files:
            if (output_handles is None):
                if not os.path.exists(decoding_path[:-4]):
                    os.mkdir(decoding_path[:-4])
                if not os.path.exists(decoding_path[:-4] + '/' +
                                      dev.dataset.task):
                    os.mkdir(decoding_path[:-4] + '/' + dev.dataset.task)
                data_name = args.test_set if args.decode_test else args.dev_set

                watcher.info(decoding_path[:-4] + '/' + dev.dataset.task)
                output_handles = [
                    open(
                        decoding_path[:-4] + '/' + dev.dataset.task +
                        '/{}.{}'.format(data_name, s), 'w')
                    for s in output_flows
                ]
        else:
            if fh is None:
                fh = open(decoding_path, 'w')

    for j, dev_batch in enumerate(dev):

        start_t = time.time()

        # decoding
        dev_outputs = model(dev_batch, mode='decoding', dataflow=dataflow)
        if hasattr(dev_batch, 'id'):
            dev_outputs['id'] = dev_batch.id.cpu().tolist()

        for w in dev_outputs:
            try:
                dev_outputs[w] = [dev_outputs[w].item()]
            except Exception:
                pass

        # gather from all workers:
        if args.distributed:
            gather_dict(dev_outputs)

        if (decoding_path is not None) and (args.local_rank == 0) and (
                args.output_on_the_fly):
            for i in range(len(dev_outputs['src'])):
                for k, d in enumerate(output_flows):
                    s = dev_outputs[d][i]
                    if args.output_decoding_files:
                        print(s, file=output_handles[k], flush=True)
                    else:
                        print('[{}]\t{}'.format(d, s), file=fh, flush=True)

        for key in dev_outputs:
            if isinstance(dev_outputs[key], list):
                outputs[key] += dev_outputs[key]
            else:
                outputs[key] += [dev_outputs[key]]

        if print_out and (j < 10):
            watcher.info("{}: {}".format(
                'source', space_tokenizer.reverse(dev_outputs['src'][0])))
            watcher.info("{}: {}".format(
                'target', space_tokenizer.reverse(dev_outputs['trg'][0])))
            watcher.info("{}: {}".format(
                'decode', space_tokenizer.reverse(dev_outputs['dec'][0])))
            watcher.info('--------------------------------------------')

        info_str = 'Decoding: sentences={}'.format(sum(outputs['sents']))
        watcher.step_progress_bar(
            info_str=info_str, step=sum(dev_outputs['sents']))
        used_t = time.time() - start_t
        curr_time += used_t

    watcher.close_progress_bar()

    if (decoding_path is not None) and (args.local_rank == 0):

        # output the sequences in the end.
        if not args.output_on_the_fly:
            # reordering the decoding into the normal order.
            indexs = list(range(len(outputs['src'])))
            if 'id' in outputs:
                indexs = np.argsort(outputs['id']).tolist()

            for i in indexs:
                for k, d in enumerate(output_flows):
                    s = space_tokenizer.reverse(outputs[d][i])
                    if args.output_decoding_files:
                        print(s, file=output_handles[k], flush=True)
                    else:
                        print('[{}]\t{}'.format(d, s), file=fh, flush=True)

        # close all file
        try:
            if args.output_decoding_files:
                for f in output_handles:
                    f.close()
            else:
                fh.close()
        except Exception:
            print('Something went wrong during decoding.')

    if args.multi_width > 1:
        outputs['speed_up'] = 1.0 / np.mean(outputs['saved_time'])
        outputs['pred_len'] = np.mean(outputs['pred_acc']) * args.multi_width
        outputs['tb_data'] += [('dev/SPEEDUP', outputs['speed_up']),
                               ('dev/PREDLEN', outputs['pred_len'])]

    # tokenize + segmentation
    sources = eval_pipe(outputs['src'])
    decodes = eval_pipe(outputs['dec'])
    targets = eval_pipe(outputs['trg'])

    # metrics
    scrores = compute_scores(targets, decodes, args.metrics)
    result_str = 'The {} dev-set, {} sentences/ {:.3f}s, '.format(
        dev.dataset.task, len(outputs['src']), curr_time)
    for metric in args.metrics:
        outputs['corpus_{}'.format(metric.lower())] = scrores[metric]
        outputs['tb_data'] += [('dev/{}/{}'.format(dev.dataset.task, metric),
                                scrores[metric])]
        result_str += '{}={:.4f}, '.format(metric, scrores[metric])

    watcher.info(result_str)

    if args.output_confounding:
        target_lang = dev.dataset.task.split('-')[1]
        outputs['confound'] = 1 - np.mean(
            [watcher.match_lang(a, target_lang) for a in outputs['dec']])
        outputs['tb_data'] += [('dev/{}/Confounding'.format(dev.dataset.task),
                                outputs['confound'])]
        watcher.info(
            "The {} dev-set outputs wrong languages with percentage = {} / {} sentences"
            .format(dev.dataset.task, outputs['confound'],
                    len(outputs['src'])))

    # clean cached memory
    torch.cuda.empty_cache()

    # output segmented data
    outputs['src'], outputs['trg'], outputs['dec'] = sources, targets, decodes
    return outputs
