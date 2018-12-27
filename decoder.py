import torch
import numpy as np
import math
import time

from collections import defaultdict
from tqdm import tqdm, trange
from utils import *

def valid_model_ppl(args, watcher, model, dev, dataflow=['src', 'trg'], lm_only=False):

    model.eval()
    outputs = defaultdict(lambda:[])
    watcher.set_progress_bar(len(dev.dataset))

    for j, dev_batch in enumerate(dev):
        dev_outputs = model(dev_batch, mode='train', dataflow=dataflow, lm_only=lm_only)
        for key in dev_outputs:
            dev_outputs[key] = [dev_outputs[key].item()]

        if args.distributed:
            gather_dict(dev_outputs)

        for key in dev_outputs:
            if isinstance(dev_outputs[key], list):
                outputs[key] += dev_outputs[key]
            else:
                outputs[key] += [dev_outputs[key]]

        info_str = 'Valid: sentences={}, ppl={:.3f}'.format(sum(outputs['sents']), np.mean(outputs['loss']))
        watcher.step_progress_bar(info_str=info_str, step=sum(dev_outputs['sents']))   

    watcher.close_progress_bar()
    
    # record for tensorboard:
    outputs['tb_data'] += [ ('dev/{}/PPL'.format(dev.dataset.task), np.mean(outputs['loss']))]
    return outputs

def valid_model(args, watcher, model, dev, dataflow=['src', 'trg'], print_out=False, decoding_path=None):

    model.eval()

    outputs = defaultdict(lambda:[])
    watcher.set_progress_bar(len(dev.dataset))

    curr_time = 0
    tokenizer = dechar if ((args.trg == 'ja') or (args.trg == 'zh')) else debpe
    segmenter = seg_kytea if (args.trg == 'ja') else (lambda x: x)
    src_tokenizer = dechar if ((args.src == 'ja') or (args.src == 'zh')) else debpe
    src_segmenter = seg_kytea if (args.src == 'ja') else (lambda x: x)
    dev.noise_flow = dataflow[0]
    
    for j, dev_batch in enumerate(dev):

        start_t = time.time()
        # decoding
        dev_outputs = model(dev_batch, mode='decoding', reverse=True, dataflow=dataflow)

        # compute sentence-level GLEU score 
        # dev_outputs['gleu'] = computeGLEU(dev_outputs['dec'], dev_outputs['trg'], corpus=False, tokenizer=tokenizer, segmenter=segmenter)
        dev_outputs['sents']  = [dev_outputs['sents'].item()]
        dev_outputs['tokens'] = [dev_outputs['tokens'].item()]
        dev_outputs['max_att'] = [dev_outputs['max_att'].item()]

        # gather from all workers:
        if args.distributed:
            gather_dict(dev_outputs)

        for key in dev_outputs:
            if isinstance(dev_outputs[key], list):
                outputs[key] += dev_outputs[key]
            else:
                outputs[key] += [dev_outputs[key]]

        if print_out and (j < 10):
            watcher.info("{}: {}".format('source', dev_outputs['src'][0]))
            watcher.info("{}: {}".format('target', dev_outputs['trg'][0]))

            if args.multi_width > 1:
                watcher.info("{}: {}".format('decode', colored_seq(dev_outputs['dec'][0], dev_outputs['decisions'][0])))
            else:
                watcher.info("{}: {}".format('decode', dev_outputs['dec'][0]))
            watcher.info('------------------------------------------------------------------')

        info_str = 'Decoding: sentences={}'.format(sum(outputs['sents']))
        
        if args.multi_width > 1:
            info_str += ', speed-up={:.4f}, pred-len={:.4f}'.format(1 / (np.mean(outputs['saved_time'])), np.mean(outputs['pred_acc']) * args.multi_width)

        watcher.step_progress_bar(info_str=info_str, step=sum(dev_outputs['sents']))    
        used_t = time.time() - start_t
        curr_time += used_t

    watcher.close_progress_bar()

    if args.multi_width > 1:
        outputs['speed_up'] = 1.0 / np.mean(outputs['saved_time'])
        outputs['pred_len'] = np.mean(outputs['pred_acc']) * args.multi_width
        outputs['tb_data'] += [('dev/SPEEDUP', outputs['speed_up']), ('dev/PREDLEN', outputs['pred_len'])]

    # tokenize + segmentation
    sources = src_segmenter([src_tokenizer(i) for i in outputs['src']])
    decodes = segmenter([tokenizer(o) for o in outputs['dec']])
    targets = segmenter([tokenizer(t) for t in outputs['trg']])

    # if not args.original:
    #     outputs['src'] = [' '.join(s) if len(s) > 0 else '--EMPTY--' for s in sources ]
    #     outputs['trg'] = [' '.join(t) if len(t) > 0 else '--EMPTY--' for t in targets ]
    #     outputs['dec'] = [' '.join(d) if len(d) > 0 else '--EMPTY--' for d in decodes ]

    outputs['corpus_bleu'] = corpus_bleu([[t] for t in targets], [o for o in decodes], emulate_multibleu=True)
    watcher.info("The dev-set corpus BLEU = {} / {} sentences".format(outputs['corpus_bleu'], len(outputs['src'])))
    
    # record for tensorboard:
    outputs['tb_data'] += [('dev/{}/BLEU'.format(dev.dataset.task), outputs['corpus_bleu'])]

    # output the sequences
    if (decoding_path is not None) and (args.local_rank == 0):
        with open(decoding_path, 'w') as fh:
            output_flows = ['src', 'trg', 'dec']
            if 'ori' in outputs:
                output_flows += ['ori']

            for i in range(len(outputs['src'])):
                for d in output_flows:
                    s = outputs[d][i]
                    print('[{}]\t{}'.format(d, s), file=fh, flush=True)

    # clean cached memory
    torch.cuda.empty_cache()

    # output segmented data
    outputs['src'], outputs['trg'], outputs['dec'] = sources, targets, decodes
    return outputs

    