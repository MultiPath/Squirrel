import torch
import numpy as np
import math
import time

from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer
from utils import *

import torch.distributed as dist


def valid_model(args, watcher, model, dev, print_out=False):

    outputs = defaultdict(lambda:[])
    model.eval()
    
    watcher.set_progress_bar(len(dev.dataset))
    for j, dev_batch in enumerate(dev):

        dev_outputs = model(dev_batch, decoding=True, reverse=True)
        
        # compute sentence-level GLEU score 
        dev_outputs['gleu'] = computeGLEU(dev_outputs['dec'], dev_outputs['trg'], corpus=False, tokenizer=debpe)
        dev_outputs['sents']  = [dev_outputs['sents'].item()]
        dev_outputs['tokens'] = [dev_outputs['tokens'].item()]

        # gather from all workers:
        if args.distributed:
            gather_dict(dev_outputs)

        for key in dev_outputs:
            if isinstance(dev_outputs[key], list):
                outputs[key] += dev_outputs[key]
            else:
                outputs[key] += [dev_outputs[key]]

        if print_out and (j < 5):
            watcher.info("{}: {}".format('source', dev_outputs['src'][0]))
            watcher.info("{}: {}".format('target', dev_outputs['trg'][0]))
            watcher.info("{}: {}".format('decode', dev_outputs['dec'][0]))
            watcher.info('------------------------------------------------------------------')

        info_str = 'Validation: sentences={}, gleu={:.3f}'.format(sum(outputs['sents']), np.mean(outputs['gleu']))
        info_str += ', speed-up={:.4f}, pred-len={:.4f}'.format(1 / (np.mean(outputs['saved_time'])), np.mean(outputs['pred_acc']) * args.multi_width)

        watcher.step_progress_bar(info_str=info_str, step=sum(dev_outputs['sents']))
        
    watcher.close_progress_bar()

    outputs['speed_up'] = 1.0 / np.mean(outputs['saved_time'])
    outputs['pred_len'] = np.mean(outputs['pred_acc']) * args.multi_width
    outputs['corpus_bleu'] = computeBLEU(outputs['dec'], outputs['trg'], corpus=True, tokenizer=debpe)
    watcher.info("The dev-set corpus BLEU = {}".format(outputs['corpus_bleu']))
    
    # record for tensorboard:
    outputs['tb_data'] = [('dev/BLEU', outputs['corpus_bleu']), 
                          ('dev/GLEU', np.mean(outputs['gleu'])),
                          ('dev/SPEEDUP', outputs['speed_up']),
                          ('dev/PREDLEN', outputs['pred_len'])]

    return outputs


def train_model(args, watcher, model, train, dev, save_path=None, maxsteps=None):

    # optimizer
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
    else:
        raise NotImplementedError

    # if resume training
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.local_rank):   # very important.
            offset, opt_states = torch.load(args.workspace_prefix + '/models/' + args.load_from + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0
    
    iters = offset

    # confirm the saving path
    if save_path is None:
        save_path = args.model_name

    # setup a watcher
    param_to_watch = ['corpus_bleu', 'speed_up', 'pred_len']
    watcher.set_progress_bar(args.eval_every)
    watcher.set_best_tracker(model, opt, save_path, args.local_rank, *param_to_watch)
    if args.tensorboard and (not args.debug):
        watcher.set_tensorboard('{}/runs/{}'.format(args.workspace_prefix, args.prefix+args.hp_str))
    
    train = iter(train)
    
    while True:

        # --- saving --- #
        if False: # (iters % args.save_every == 0) and (args.local_rank == 0): # saving only works for local-rank=0
            # watcher.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.local_rank):
                torch.save(watcher.best_tracker.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
                torch.save([iters, watcher.best_tracker.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))

        # --- validation --- #
        if (iters % args.eval_every == 10000000): # and (args.local_rank == 0):

            watcher.close_progress_bar()

            with torch.no_grad():
                outputs_data = valid_model(args, watcher, model, dev, print_out=False)

            if args.tensorboard and (not args.debug):
                if len(outputs_data['tb_data']) > 0:
                    for name, value in outputs_data['tb_data']:
                        watcher.add_tensorboard(name, value, iters)

            if not args.debug:
                watcher.acc_best_tracker(iters, outputs_data['corpus_bleu'], outputs_data['speed_up'], outputs_data['pred_len'])
                # print(outputs_data['corpus_bleu'], outputs_data['speed_up'], outputs_data['pred_len'])

                if args.local_rank == 0:
                    watcher.info('the best model is achieved at {}, corpus BLEU={}/speed-up={:.4f}/pred-len={:.4f}'.format(watcher.best_tracker.i, 
                                    watcher.best_tracker.corpus_bleu, watcher.best_tracker.speed_up, watcher.best_tracker.pred_len))
            watcher.info('model:' + args.prefix + args.hp_str)


            # ---set-up a new progressor---
            watcher.set_progress_bar(args.eval_every)

        if maxsteps is None:
            maxsteps = args.maximum_steps

        if iters > maxsteps:
            watcher.info('reach the maximum updating steps.')
            break
        

        # --- training  --- #
        iters += 1
        model.train()

        def get_learning_rate(i, disable=False):
            if not disable:
                return min(max(1.0 / math.sqrt(args.d_model * i), 5e-5), i / (args.warmup * math.sqrt(args.d_model * args.warmup)))               
            return 0.00002

        with Timer() as train_timer:
        
            opt.param_groups[0]['lr'] = get_learning_rate(iters, disable=args.disable_lr_schedule)
            opt.zero_grad()
                
            info_str = 'training step = {}, lr={:.7f}, '.format(iters, opt.param_groups[0]['lr'])
            info = defaultdict(lambda:[])

            # prepare the data
            for inter_step in range(args.inter_size):

                batch = next(train)  # load the next batch of training data.
                info_ = model(batch)
                info_['loss'] = info_['loss'] / args.inter_size
                info_['loss'].backward()

                for t in info_:
                    info[t] += [info_[t].item()]
                
            # multiple steps, one update
            opt.step()

            if args.distributed:  # gather information from other workers.
                gather_dict(info)
            
            for t in info:
                info[t] = sum(info[t])

        info_str += '{} tokens / batch, {} tokens / sec, '.format(
                    int(info['tokens']), 
                    int(info['tokens'] / train_timer.elapsed_secs))

        info_str += 'MLE_loss={:.3f}, '.format(info['MLE'] / args.world_size / args.inter_size)
        if args.encoder_lm and args.causal_enc:
            info_str += 'ENCLM_loss={:.3f}, '.format(info['LM'] / args.world_size / args.inter_size)

        if args.tensorboard and (not args.debug):
            watcher.add_tensorboard('train/Loss', info['MLE'] / args.world_size / args.inter_size, iters)
            if args.encoder_lm and args.causal_enc:
                watcher.add_tensorboard('train/Enc_LM_loss', info['LM'] / args.world_size / args.inter_size, iters)
        
        watcher.step_progress_bar(info_str=info_str)
