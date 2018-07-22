import torch
import numpy as np
import math
import time

from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer
from utils import Metrics, Best, computeGLEU, computeBLEU

# -- helper functions

def export(x):
    try:
        with torch.cuda.device_of(x):
            return x.data.cpu().float().mean()
    except Exception:
        return 0

def debpe(x):
    return x.replace('@@ ', '').split()

def valid_model(args, model, dev, print_out=False):
    print_seqs = ['[sources]', '[targets]', '[decoded]']
    outputs = {'src': [], 'trg': [], 'dec': [], 'gleu': []}

    model.eval()

    for j, dev_batch in enumerate(dev):
        with torch.no_grad():

            # prepare the data
            source_inputs, source_outputs, source_masks, \
            target_inputs, target_outputs, target_masks = model.prepare_data(dev_batch)

            # encoding
            encoding_outputs = model.encoding(source_inputs, source_masks)

            # decoding
            decoding_outputs, out, probs = model(encoding_outputs, source_masks, target_inputs, target_masks, 
                                                decoding=True, return_probs=True)
            
            # reverse to string-sequence
            dev_outputs = [('src', source_outputs), ('trg', target_outputs), ('trg', decoding_outputs)]
            dev_outputs = [model.output_decoding(d) for d in dev_outputs]
            
            # compute sentence-level GLEU score 
            gleu = computeGLEU(dev_outputs[2], dev_outputs[1], corpus=False, tokenizer=debpe)
            
            # save to the outputs
            outputs['src'] += dev_outputs[0]
            outputs['trg'] += dev_outputs[1]
            outputs['dec'] += dev_outputs[2]
            outputs['gleu'].append(gleu)


            if print_out:
                for k, d in enumerate(dev_outputs):
                    args.logger.info("{}: {}".format(print_seqs[k], d[0]))
                args.logger.info('------------------------------------------------------------------')

    outputs['corpus_bleu'] = computeBLEU(outputs['dec'], outputs['trg'], corpus=True, tokenizer=debpe)
    args.logger.info("The dev-set corpus BLEU = {}".format(outputs['corpus_bleu']))
    return outputs


def train_model(args, model, train, dev, save_path=None, maxsteps=None):

    # record by tensorbard.
    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('{}/runs/{}'.format(args.workspace_prefix, args.prefix+args.hp_str))

    # optimizer
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
    else:
        raise NotImplementedError

    # if resume training
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.gpu):   # very important.
            offset, opt_states = torch.load(args.workspace_prefix + '/models/' + args.load_from + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0

    # metrics
    if save_path is None:
        save_path = args.model_name

    iters = offset
    best = Best(max, 'corpus_bleu', 'sentence_gleu', 'i', model=model, opt=opt, path=save_path, gpu=args.gpu)
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    # statistics
    total_tokens = 0

    while True:

        # --- saving --- #
        if iters % args.save_every == 0:
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
                torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))

        # --- validation --- #
        if False: #iters % args.eval_every == 0:
            progressbar.close()

            outputs_data = valid_model(args, model, dev, print_out=True)
            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/GLEU_sentence_', np.mean(outputs_data['gleu']), iters)
                # writer.add_scalar('dev/Loss', dev_metrics.loss, iters)
                writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], iters)

            if not args.debug:
                best.accumulate(outputs_data['corpus_bleu'], np.mean(outputs_data['gleu']), iters)
                args.logger.info('the best model is achieved at {}, average greedy GLEU={}, corpus BLEU={}'.format(best.i, best.sentence_gleu, best.corpus_bleu))
            args.logger.info('model:' + args.prefix + args.hp_str)

            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')


        if maxsteps is None:
            maxsteps = args.maximum_steps

        if iters > maxsteps:
            args.logger.info('reach the maximum updating steps.')
            break

        # --- training  --- #
        iters += 1
        model.train()

        def get_learning_rate(i, lr0=0.1, disable=False):
            if not disable:
                return lr0 * 10 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
            return 0.00002

        opt.param_groups[0]['lr'] = get_learning_rate(iters + 1, disable=args.disable_lr_schedule)
        opt.zero_grad()
        
        info_str = 'training step = {}, lr={:.5f}, '.format(iters, opt.param_groups[0]['lr'])
        info = defaultdict(lambda:0)

        # prepare the data
        for inter_step in range(args.inter_size):

            batch = next(iter(train))  # load the next batch of training data.

            source_inputs, source_outputs, source_masks, \
            target_inputs, target_outputs, target_masks = model.prepare_data(batch)
            
            info['sents']  += target_inputs.size(0)
            info['tokens'] += (target_masks != 0).data.sum()
            
            # encoding
            encoding_outputs = model.encoding(source_inputs, source_masks)

            # Maximum Likelihood Training (with label smoothing trick)
            decoding_outputs = model(encoding_outputs, source_masks, target_inputs, target_masks)        
            loss  = model.cost(target_outputs, target_masks, out=decoding_outputs, label_smooth=args.label_smooth)
            info['MLE'] += export(loss)

            # Source side Language Model (optional, only works for causal-encoder)
            if args.encoder_lm and args.causal_enc:
                loss_lm = model.cost(source_outputs, source_masks, out=encoding_outputs[-1])
                info['LM'] += export(loss_lm)
                loss += loss_lm

            loss = loss / args.inter_size
            loss.backward()
            
        # multiple steps, one update
        opt.step()

        info_str += 'Processed {} sents/{} tokens'.format(info['sents'], info['tokens'])
        info_str += 'MLE_loss={:.3f}, '.format(info['MLE'] / args.inter_size)
        if args.encoder_lm and args.causal_enc:
            info_str += 'ENCLM_loss={:.3f}, '.format(info['LM'] / args.inter_size)

        if args.tensorboard and (not args.debug):
            writer.add_scalar('train/Loss', export(loss), iters)

        total_tokens += info['tokens']
        progressbar.update(1)
        progressbar.set_description(info_str)

