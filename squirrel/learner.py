import math
from collections import defaultdict
from subprocess import call

import numpy as np
import torch

from squirrel.data.batch import merge_batches, split_batch
from squirrel.decoder import valid_model, valid_model_ppl
from squirrel.optimizer import Adam
from squirrel.utils import Timer, format, gather_dict, item

# class AsynchronousPreprocessing(object):
#     def __init__(self, train, model):
#         self.train_iter = [iter(t) for t in train]
#         self.model = model


def get_learning_rate(args, i):
    if not args.disable_lr_schedule:
        if args.lr == 0:  # use pre-defined learning rate
            return min(
                max(1.0 / math.sqrt(args.d_model * i), 5e-5),
                i / (args.warmup * math.sqrt(args.d_model * args.warmup)))

        else:
            # manually define the learning rate (the same as fairseq-py)
            warmup_end_lr = args.lr
            warmup_init_lr = 1e-7
            lr_step = (warmup_end_lr - warmup_init_lr) / args.warmup
            decay_factor = warmup_end_lr * args.warmup**0.5

            if i < args.warmup:
                return warmup_init_lr + i * lr_step
            else:
                return decay_factor * (i**-0.5)
    return 0.001


def train_model(args,
                watcher,
                model,
                train,
                dev,
                save_path=None,
                maxsteps=None,
                decoding_path=None,
                names=None):

    # optimizer
    all_opt = [
        Adam(
            param, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
        for param in model.module.trainable_parameters()
    ]
    opt = all_opt[0]

    # if resume training
    if (args.load_from != 'none') and (args.resume):
        with torch.cuda.device(args.local_rank):  # very important.
            offset, opt_states = torch.load(
                args.workspace_prefix + '/models/' + args.load_from +
                '.pt.states',
                map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0

    iters, best_i = offset, 0

    # confirm the saving path
    if save_path is None:
        save_path = args.model_name

    # setup a watcher
    param_to_watch = ['corpus_bleu']
    watcher.set_progress_bar(args.eval_every)
    watcher.set_best_tracker(model, opt, save_path, args.local_rank,
                             *param_to_watch)
    if args.tensorboard and (not args.debug):
        watcher.set_tensorboard('{}/runs/{}'.format(args.workspace_prefix,
                                                    args.prefix + args.hp_str))
        watcher.add_tensorboard(
            'Hyperparameters', '  \n'.join([
                '{}:\t{}'.format(a, b)
                for a, b in sorted(args.__dict__.items(), key=lambda x: x[0])
            ]), 0, 'text')

    train_iter = [iter(t) for t in train]
    while True:

        def check(every=0, k=0):
            if every <= 0:
                return False
            return iters % every == k

        # whether we only train LM or not ---> task
        if ((iters < args.lm_steps) or (args.lm_only)):
            task = 'lm_only'
        else:
            task = None

        # --- saving --- #
        if check(args.save_every) and (
                args.local_rank == 0):  # saving only works for local-rank=0
            watcher.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.local_rank):
                torch.save(watcher.best_tracker.model.state_dict(),
                           '{}_iter={}.pt'.format(args.model_name, iters))
                torch.save(
                    [iters, watcher.best_tracker.opt.state_dict()],
                    '{}_iter={}.pt.states'.format(args.model_name, iters))

        # --- validation --- #
        if check(args.eval_every):  # and (args.local_rank == 0):
            if args.local_rank == 0:
                call(["hostname"])
                call([
                    "nvidia-smi", "--format=csv",
                    "--query-gpu=memory.used,memory.free"
                ])

            watcher.close_progress_bar()
            if not args.no_valid:
                with torch.no_grad():
                    outputs_data = [
                        valid_model(
                            args,
                            watcher,
                            model,
                            d,
                            print_out=True,
                            dataflow=['src', 'trg']) for d in dev
                    ]

                if args.tensorboard and (not args.debug):
                    for outputs in outputs_data:
                        for name, value in outputs['tb_data']:
                            watcher.add_tensorboard(name, value, iters)

                if not args.debug:
                    if len(outputs_data) == 1:  # single pair MT
                        avg_corpus_bleu = outputs_data[0]['corpus_bleu']
                        requires_tracking = [0]

                        sources = outputs_data[0]['src']
                        decodes = outputs_data[0]['dec']
                        targets = outputs_data[0]['trg']

                    else:
                        # for multilingual training, we need to compute the overall BLEU
                        # which is merge the dataset and re-evaluate
                        if args.track_best is not None:
                            requires_tracking = [
                                int(a) for a in args.track_best.split(',')
                            ]
                        else:
                            requires_tracking = list(range(len(dev)))

                        sources, decodes, targets = [], [], []
                        for i in requires_tracking:
                            sources += outputs_data[i]['src']
                            decodes += outputs_data[i]['dec']
                            targets += outputs_data[i]['trg']

                        avg_corpus_bleu = np.mean([
                            outputs_data[i]['corpus_bleu']
                            for i in requires_tracking
                        ])

                    if args.tensorboard and (not args.debug):
                        if len(outputs_data) > 1:
                            watcher.add_tensorboard('dev/average_BLEU',
                                                    avg_corpus_bleu, iters)

                        L = len(sources)
                        txt = ''
                        for i in range(10, L, L // 8):
                            txt += 'source:     ' + ' '.join(
                                sources[i]) + '  \n'
                            txt += 'target:     ' + ' '.join(
                                targets[i]) + '  \n'
                            txt += 'decode:     ' + ' '.join(
                                decodes[i]) + '  \n'
                            txt += '-----------  \n'
                        watcher.add_tensorboard(
                            'Translation sample', txt, iters, dtype='text')
                    watcher.acc_best_tracker(iters, avg_corpus_bleu)

                    if args.test_src is not None:
                        test_srcs, test_trgs = args.test_src.split(
                            ','), args.test_trg.split(',')
                    else:
                        test_srcs, test_trgs = args.src.split(
                            ','), args.trg.split(',')
                    watcher.info('tracking for language pairs: {}'.format(
                        '/'.join([
                            '{}-{}'.format(test_srcs[i], test_trgs[i])
                            for i in requires_tracking
                        ])))
                    watcher.info(
                        'the best model is achieved at {}, corpus BLEU={}'.
                        format(watcher.best_tracker.i,
                               watcher.best_tracker.corpus_bleu))

                    if args.local_rank == 0:
                        if watcher.best_tracker.i > best_i:
                            best_i = watcher.best_tracker.i

                watcher.info('model:' + args.prefix + args.hp_str)

            if args.valid_ppl:
                with torch.no_grad():
                    outputs_data = [
                        valid_model_ppl(
                            args,
                            watcher,
                            model,
                            d,
                            dataflow=['src', 'trg'],
                            task=task) for d in dev
                    ]

                if args.tensorboard and (not args.debug):
                    for outputs in outputs_data:
                        for name, value in outputs['tb_data']:
                            watcher.add_tensorboard(name, value, iters)
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

        info_str = 'training step = {}, lr={:.7f}, '.format(
            iters, opt.param_groups[0]['lr'])
        info = defaultdict(lambda: [])
        pairs = []

        with Timer() as train_timer:

            opt.param_groups[0]['lr'] = get_learning_rate(
                args, iters)  # (args.model == 'AutoTransformer2'))
            opt.zero_grad()

            # prepare the data
            for inter_step in range(args.inter_size):

                def sample_a_training_set(train, prob):
                    if (prob is None) or (
                            len(prob) == 0
                    ):  # not providing probability, sample dataset uniformly.
                        prob = [1 / len(train) for _ in train]
                    train_idx = np.random.choice(np.arange(len(train)), p=prob)
                    return next(train[train_idx])

                def merge_training_sets(train):
                    return merge_batches([next(train_i) for train_i in train])

                if len(train) == 1:  # single-pair MT:
                    batch = next(train_iter[0])

                else:
                    if args.sample_a_training_set:
                        batch = sample_a_training_set(train_iter,
                                                      args.sample_prob)
                    else:
                        batch = merge_training_sets(train_iter)

                # --- attention visualization --- #
                if (check(args.att_plot_every, 1) and (inter_step == 0)
                        and (args.local_rank == 0)):
                    model.module.attention_flag = True

                # -- search optimal paths (for training insertable transformer) -- #
                if (args.order == 'random') or (args.order == 'optimal'):

                    if args.search_with_dropout:
                        model.train()
                    else:
                        model.eval()

                    with torch.no_grad():
                        infob_ = model(
                            batch,
                            mode='path',
                            dataflow=['src', 'trg'],
                            step=iters)
                        for t in infob_:
                            info[t] += [item(infob_[t])]

                # training with dropout
                model.train()  # open drop-out
                DIV = args.inter_size * args.sub_inter_size

                for batch_ in split_batch(batch, args.sub_inter_size):
                    mode = 'search_train' if args.order == 'search_optimal' else 'train'
                    info_ = model(
                        batch_, mode=mode, dataflow=['src', 'trg'], step=iters)
                    info_['loss'] = info_['loss'] / DIV
                    info_['loss'].backward()

                    pairs.append(batch.task + batch.message)
                    for t in info_:
                        info[t] += [item(info_[t])]

            # multiple steps, one update
            grad_norm = opt.clip_grad_norm(args.grad_clip)
            opt.step()

            if args.distributed:  # gather information from other workers.
                gather_dict(info)

            for t in info:
                try:
                    info[t] = sum(info[t])
                except TypeError:
                    continue

        if check(args.print_every) and (args.order != 'fixed'):
            watcher.info("--------" * 15)
            for s in range(min(3, len(info['src']))):
                watcher.info("{}:\t{}".format('source', info['src'][s]))
                watcher.info("{}:\t{}".format('target', info['trg'][s]))
                if 'reorder' in info:
                    watcher.info("{}:\t{}".format('reorder',
                                                  info['reorder'][s]))
                watcher.info("--------" * 15)

        # basic infomation
        info_str += '#sentence={}, #token={}, '.format(
            int(info['sents']), format(info['tokens'], 'k'))
        if 'full_tokens' in info:
            info_str += '#token(F)={}, '.format(
                format(info['full_tokens'], 'k'))

        info_str += 'gn={:.4f}, speed={} t/s | BEST={} | '.format(
            grad_norm, format(info['tokens'] / train_timer.elapsed_secs, 'k'),
            watcher.best_tracker.corpus_bleu)

        for keyword in info:
            if keyword[:2] == 'L@':
                info_str += '{}={:.3f}, '.format(
                    keyword, info[keyword] / args.world_size / DIV)
                if args.tensorboard and (not args.debug):
                    watcher.add_tensorboard(
                        'train/{}'.format(keyword),
                        info[keyword] / args.world_size / DIV, iters)

        if args.tensorboard and (not args.debug):
            watcher.add_tensorboard('train/LR', opt.param_groups[0]['lr'],
                                    iters)
            # -- attention visualization -- #
            if (model.module.attention_maps is
                    not None) and (args.local_rank == 0):
                watcher.info('Attention visualization at Tensorboard')
                with Timer() as visualization_timer:
                    for name, value in model.module.attention_maps:
                        watcher.add_tensorboard(name, value, iters, 'figure')
                    model.module.attention_maps = None
                watcher.info('Attention visualization cost: {}s'.format(
                    visualization_timer.elapsed_secs))

        watcher.step_progress_bar(info_str=info_str)
