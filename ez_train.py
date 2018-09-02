import torch
import numpy as np
import math
import time
import time

from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer
from utils import Metrics, Best, computeGLEU, computeBLEU, export, debpe
from utils import gather_tensor, reduce_tensor, reduce_dict

import torch.distributed as dist


def valid_model(args, model, dev, print_out=False):

    print_seqs = ['[sources]', '[targets]', '[decoded]']
    outputs = defaultdict(lambda:[])

    progressbar = tqdm(total=len([1 for _ in dev]), desc='start decoding for validation...')
    model.eval()

    for j, dev_batch in enumerate(dev):
       
        # prepare the data
        source_inputs, source_outputs, source_masks, \
        target_inputs, target_outputs, target_masks = model.prepare_data(dev_batch)

        # encoding
        encoding_outputs = model.encoding(source_inputs, source_masks)
        if args.causal_enc and args.encoder_lm:
            outputs['accuracy'] += model.io_enc.acc(source_outputs, source_masks, encoding_outputs[-1])
        
        # decoding
        decoding_outputs, out, probs = model.decoding(encoding_outputs, source_masks, target_inputs, target_masks, 
                                                      decoding=True, return_probs=True)
        
        # reverse to string-sequence
        dev_outputs = [model.io_enc.reverse(source_outputs),
                       model.io_dec.reverse(target_outputs),
                       model.io_dec.reverse(decoding_outputs)]
        
        # compute sentence-level GLEU score 
        gleu = computeGLEU(dev_outputs[2], dev_outputs[1], corpus=False, tokenizer=debpe)
        
        # save to the outputs
        outputs['src'] += dev_outputs[0]
        outputs['trg'] += dev_outputs[1]
        outputs['dec'] += dev_outputs[2]
        outputs['gleu'] += gleu

        if print_out and (j < 5):
            for k, d in enumerate(dev_outputs):
                args.logger.info("{}: {}".format(print_seqs[k], d[0]))
                
            args.logger.info('------------------------------------------------------------------')

        info = 'Validation: decoding step={}, gleu={:.3f}'.format(j + 1, np.mean(outputs['gleu']))
        if args.causal_enc and args.encoder_lm:
            info += ', source predict acc={:.3f}'.format(np.mean(outputs['accuracy']))

        progressbar.update(1)
        progressbar.set_description(info)
    progressbar.close()

    outputs['corpus_bleu'] = computeBLEU(outputs['dec'], outputs['trg'], corpus=True, tokenizer=debpe)
    args.logger.info("The dev-set corpus BLEU = {}".format(outputs['corpus_bleu']))
    return outputs


def train_model(args, model, train, dev, save_path=None, maxsteps=None):

    # record by tensorbard.
    if args.tensorboard and (not args.debug) and (args.local_rank == 0):
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

    if args.local_rank == 0:
        best = Best(max, 'corpus_bleu', 'sentence_gleu', 'i', model=model, opt=opt, path=save_path, gpu=args.gpu)
        progressbar = tqdm(total=args.eval_every, desc='start training.')

    # statistics
    total_tokens = 0
    train = iter(train)
    
    # save the arguments
    # torch.save(args, '{}.pt.options'.format(args.model_name))

    while True:

        # --- saving --- #
        if (iters % args.save_every == 1) and (args.local_rank == 0): # saving only works for local-rank=0
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
                torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))

        # --- validation --- #
        if (iters % args.eval_every == 1) and (args.local_rank == 0): # validation only works for local-rank=0
            progressbar.close()

            with torch.no_grad():
                outputs_data = valid_model(args, model.module if args.distributed else model, dev, print_out=True)

            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/GLEU_sentence_', np.mean(outputs_data['gleu']), iters)
                # writer.add_scalar('dev/Loss', dev_metrics.loss, iters)
                writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], iters)
                
                if args.causal_enc and args.encoder_lm:
                    writer.add_scalar('dev/Source_Predict_', np.mean(outputs_data['accuracy']), iters)

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

        def get_learning_rate(i, disable=False):
            if not disable:
                return min(max(1.0 / math.sqrt(args.d_model * i), 5e-5), i / (args.warmup * math.sqrt(args.d_model * args.warmup)))
                # return 10 * lr0 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
            return 0.00002

        opt.param_groups[0]['lr'] = get_learning_rate(iters, disable=args.disable_lr_schedule)
        opt.zero_grad()
        
        
        info_str = 'training step = {}, lr={:.7f}, '.format(iters, opt.param_groups[0]['lr'])
        info = defaultdict(lambda:0)

        # prepare the data
        for inter_step in range(args.inter_size):
            t0 = time.time()
            batch = next(train)  # load the next batch of training data.
            t1 = time.time()
            
            loss, info_ = model(batch, info)
            loss = loss / args.inter_size
            loss.backward()
            t2 = time.time()

            # print('I am rank {}, inner={}: t1={}, t2={}, size={}'.format(args.local_rank, inter_step, t1-t0, t2-t1, batch.src.size()))

        # multiple steps, one update
        opt.step()
        total_tokens += info['tokens']

        if args.distributed:  # gather information from other workers.
            reduce_dict(info)


        info = export(info)
        info_str += '{} sents/{} tokens, total {} tokens, '.format(int(info['sents']), int(info['tokens']), format(total_tokens, ','))
        info_str += 'MLE_loss={:.3f}, '.format(info['MLE'] / args.inter_size / args.world_size)
        if args.encoder_lm and args.causal_enc:
            info_str += 'ENCLM_loss={:.3f}, '.format(info['LM'] / args.inter_size / args.world_size)

        if args.tensorboard and (not args.debug) and (args.local_rank == 0):
            writer.add_scalar('train/Loss', info['MLE'] / args.inter_size, iters)
            if args.encoder_lm and args.causal_enc:
                writer.add_scalar('train/Enc_LM_loss', info['LM'] / args.inter_size)
        
        if args.local_rank == 0:
            progressbar.update(1)
            progressbar.set_description(info_str)

