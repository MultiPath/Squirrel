import os
import torch
import numpy as np
import time

from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, INF, TINY, softmax
from utils import NormalField, ParallelDataset
from utils import Metrics, Best, computeGLEU, computeBLEU, Batch, masked_sort
from time import gmtime, strftime


debpe = lambda x: x.replace('@@ ', '').split()

def cutoff(s, t):
    for i in range(len(s), 0, -1):
        if s[i-1] != t:
            return s[:i]
    print(s)
    raise IndexError


def decode_model(args, model, dev, evaluate=True, decoding_path=None, names=None, maxsteps=None):

    args.logger.info("decoding with {}, beam_size={}, alpha={}".format(args.decode_mode, args.beam_size, args.alpha))
    dev.train = False  # make iterator volatile=True

    if maxsteps is None:
        maxsteps = sum([1 for _ in dev])
    progressbar = tqdm(total=maxsteps, desc='start decoding')

    model.eval()
    if decoding_path is not None:
        handles = [open(os.path.join(decoding_path, name), 'w') for name in names]

    corpus_size = 0
    src_outputs, trg_outputs, dec_outputs, timings = [], [], [], []
    decoded_words, target_words, decoded_info = 0, 0, 0

    attentions = None
    pad_id = model.decoder.field.vocab.stoi['<pad>']
    eos_id = model.decoder.field.vocab.stoi['<eos>']

    curr_time = 0
    for iters, dev_batch in enumerate(dev):

        if iters > maxsteps:
            args.logger.info('complete {} steps of decoding'.format(maxsteps))
            break

        start_t = time.time()

        # prepare the data
        source_inputs, source_outputs, source_masks, \
        target_inputs, target_outputs, target_masks = model.prepare_data(dev_batch)
        
        # encoding
        encoding_outputs = model.encoding(source_inputs, source_masks)

        # decoding
        decoding_outputs = model.decoding(encoding_outputs, source_masks, target_inputs, target_masks, 
                                        beam=args.beam_size, alpha=args.alpha, decoding=True, return_probs=False)
        
        # reverse to string-sequence
        dev_outputs = [('src', source_outputs), ('trg', target_outputs), ('trg', decoding_outputs)]
        dev_outputs = [model.output_decoding(d) for d in dev_outputs]

        used_t = time.time() - start_t
        curr_time += used_t

        real_mask = 1 - ((decoding_outputs == eos_id) + (decoding_outputs == pad_id)).float()

        corpus_size += source_inputs.size(0)
        src_outputs += dev_outputs[0]
        trg_outputs += dev_outputs[1]
        dec_outputs += dev_outputs[2]
        timings += [used_t]

        if decoding_path is not None:
            for s, t, d in zip(dev_outputs[0], dev_outputs[1], dev_outputs[2]):
                if args.no_bpe:
                    s, t, d = s.replace('@@ ', ''), t.replace('@@ ', ''), d.replace('@@ ', '')
                print(s, file=handles[0], flush=True)
                print(t, file=handles[1], flush=True)
                print(d, file=handles[2], flush=True)

        progressbar.update(1)
        progressbar.set_description('finishing sentences={}/batches={}, speed={:.2f} sentences / sec'.format(corpus_size, iters, corpus_size / curr_time))

    if evaluate:
        corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=debpe)
        args.logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))

