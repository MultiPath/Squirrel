from .core import *
from .transformer import Transformer
import copy

class TransformerIns(Transformer):
    
    def __init__(self, src, trg, args):
        super().__init__(src, trg, args)
        self.pos_transform = Linear(args.d_model, 4 * args.d_model)
        self.pos_attention = Attention(args.d_model, 1, 0, causal=True)  # important! has to be causal #
        
        if args.ln_pos:
            self.pos_w1 = Linear(args.d_model, args.d_model)
            self.pos_w2 = Linear(args.d_model, args.d_model)

        # hyper-parameters
        self.insert_mode = args.insert_mode 
        self.order_mode = args.order
        self.name = 'original'
    
    def reverse(self, batch, pos, dir, dataflow='trg'):
        batch = self.fields[dataflow].reverse(batch, reverse_token=False)
        init_token = self.fields[dataflow].init_token
        new_batch = []

        for i, ex in enumerate(batch):
            _batch = [init_token, '<stop>'] + ex
            _seq = {k: [None, None] for k in range(len(_batch))}
            _seq[0][1] = 1
            _seq[1][0] = 0

            for j, _ in enumerate(ex):
                a = pos[i, j].item()
                d = dir[i, j].item()
                e = j + 2

                if d == 0: # insert to left
                    if _seq[a][0] is None:
                        _seq[a][0] = e
                        _seq[e][1] = a
                    else:
                        c = _seq[a][0]
                        _seq[a][0] = e
                        _seq[e][1] = a
                        _seq[e][0] = c
                        _seq[c][1] = e

                else:         # insert to right
                    if _seq[a][1] is None:
                        _seq[a][1] = e
                        _seq[e][0] = a
                    else:
                        c = _seq[a][1]
                        _seq[a][1] = e
                        _seq[e][0] = a
                        _seq[e][1] = c
                        _seq[c][0] = e

            _batch_out = []
            w = 0
            while w is not None:
                _batch_out.append(_batch[w])
                w = _seq[w][1]
            
            new_batch.append(self.fields[dataflow].reverse_tokenizer(_batch_out[1:-1]))
        return new_batch, [self.fields[dataflow].reverse_tokenizer(b) for b in batch]
    
    def pointing(self, pointer_query, pointer_key_l, pointer_key_r, 
                target_mask=None, insertion_l=None, insertion_r=None, 
                output=True, non_causal=False, weights=None, reduction='mean'):

        def add_mask(mask, i):
            _mask = mask.clone()
            _mask[:, i] = 0
            return _mask

        if target_mask is None:
            target_mask = pointer_key_l.new_ones(*pointer_key_l.size()[:2]).float()

        l_mask, r_mask = add_mask(target_mask, 0), add_mask(target_mask, 1)
        if self.args.no_bound and (l_mask.size(1) > 2):
            l_mask, r_mask = add_mask(l_mask, 1), add_mask(r_mask, 0)

        if isinstance(pointer_query, list):
            pointer_query, pointer_embed = pointer_query
            pointer_scores_l = self.pos_attention(pointer_query, pointer_key_l.detach(), value=None, mask=l_mask, non_causal=non_causal) \
                            +  self.pos_attention(pointer_embed, pointer_key_l, value=None, mask=l_mask, non_causal=non_causal)
            pointer_scores_r = self.pos_attention(pointer_query, pointer_key_r.detach(), value=None, mask=r_mask, non_causal=non_causal) \
                            +  self.pos_attention(pointer_embed, pointer_key_r, value=None, mask=r_mask, non_causal=non_causal)
        else:
            pointer_scores_l = self.pos_attention(pointer_query, pointer_key_l, value=None, mask=l_mask, non_causal=non_causal)
            pointer_scores_r = self.pos_attention(pointer_query, pointer_key_r, value=None, mask=r_mask, non_causal=non_causal)    


        pointer_scores = torch.cat([pointer_scores_l, pointer_scores_r], 2)

        if (insertion_l is not None) and (insertion_r is not None):  # given supervison.
            pointer_probs = softmax(pointer_scores)[:, 1:]           # ignore the first prediction.
            pointer_probs_l, pointer_probs_r = pointer_probs[:, :, :pointer_probs.size(2) // 2], pointer_probs[:, :, pointer_probs.size(2) // 2:]

            pointer_probs_l = pointer_probs_l.gather(2, insertion_l[:, :, None]) 
            pointer_probs_r = pointer_probs_r.gather(2, insertion_r[:, :, None])
            loss_pointer = -torch.log(pointer_probs_l + pointer_probs_r + TINY).squeeze(2)

            if reduction == 'none':
                return loss_pointer

            if weights is None:
                loss_pointer = loss_pointer[target_mask[:, 1:].byte()].mean()
            else:
                weights = weights[:, None].expand_as(loss_pointer)[target_mask[:, 1:].byte()]
                loss_pointer = (loss_pointer[target_mask[:, 1:].byte()] * weights).mean()

            if not output:
                return loss_pointer

            pointer_out_l = pointer_key_l.gather(1, insertion_l[:, :, None].expand(insertion_l.size(0), -1, pointer_key_l.size(2)))
            pointer_out_r = pointer_key_r.gather(1, insertion_r[:, :, None].expand(insertion_r.size(0), -1, pointer_key_r.size(2)))
            pointer_out = pointer_out_l * (pointer_probs_l > pointer_probs_r).float() + pointer_out_r * (pointer_probs_l <= pointer_probs_r).float()
            return loss_pointer, pointer_out

        if not output:
            return pointer_scores

        pointer_pos = pointer_scores.squeeze(1).max(1)[1]
        pointer_out = torch.cat([pointer_key_l, pointer_key_r], 1).gather(1, pointer_pos[:, None, None].expand(pointer_pos.size(0), 1, pointer_key_l.size(2))).squeeze(1)
        return pointer_pos, pointer_out

    def fusion(self, pointer_query, word_embed):
        if self.args.ln_pos:
            return [pointer_query, word_embed]
            # return self.pos_w2(gelu(pointer_query + self.pos_w1(word_embed)))
        else:
            return (pointer_query + word_embed) * (0.5 ** 0.5)  # normalize the scale
            
    def expand_stack(self, data, W=None):
        if W is not None:
            data = data[:, None, :].expand(data.size(0), W, *data.size()[1:]).contiguous().view(data.size(0) *  W, *data.size()[1:])
        else:
            data = data.view(data.size(0) *  data.size(1), *data.size()[2:])
        return data

    def generate_random_matrix(self, data, factor):
        return data.new_zeros(data.size()).uniform_()
        # scores = data.new_zeros(self.args.batch_size * self.args.world_size * factor).uniform_()
        # scores = scores[self.args.batch_size * self.args.local_rank * factor: self.args.batch_size * self.args.local_rank * factor + data.numel()].view(data.size())
        # return scores

    def forward(self, batch, mode='train', reverse=True, dataflow=['src', 'trg'], step=None, lm_only=False):
        """
        batch: examples
        mode:  train, decoding, search
        """

        #if info is None:
        info = defaultdict(lambda: 0)

        if batch.preprocessed is None:
            source_inputs, source_outputs, source_masks, \
            target_inputs, target_outputs, target_masks = self.prepare_data(batch, dataflow=dataflow)
            
            if mode != 'decoding':
                target_positions = batch.pos - 2
        
        else:
            source_inputs, source_outputs, source_masks, \
            target_inputs, target_outputs, target_masks, \
            target_positions = batch.preprocessed
        
        weights = batch.weights

        # encoding: if first dataflow is empty, then ignore the encoder.
        if not lm_only:
            encoding_inputs  = self.io_enc.i(source_inputs, pos=(not self.relative_pos))
            encoding_outputs = self.encoder(encoding_inputs, source_masks)
        else:
            encoding_outputs, source_masks = None, None

        # decoding
        if mode == 'path':
            sources = [encoding_outputs, source_masks]
            targets = [target_inputs, target_outputs, target_masks, target_positions]
            
            info['L@G'] = min(self.args.gamma, (step - self.args.esteps) / (self.args.gsteps))
            info['L@E'] = self.args.epsilon  # noise: exploration search random orders.

            if self.order_mode == 'random':
                targets = self.randomize_order(targets)

                batch.preprocessed = [source_inputs, source_outputs, source_masks] + targets
                batch.message = '(R)'

            elif self.order_mode == 'optimal':

                if self.args.sample_order:
                    gamma  = 10 - 9 * min(1, step / self.args.gsteps)
                    targets, weights = self.particle_filtering(sources, targets, self.args.beta, resampling=self.args.resampling, gamma=gamma)
                else:
                    targets, weights = self.search_optimal_order(sources, targets, self.args.beta, gamma=info['L@G'], epsilon=info['L@E'])
                
                batch.preprocessed = [source_inputs, source_outputs, source_masks] + targets
                batch.weights = weights
                batch.message = '(O)'

            else:
                raise NotImplementedError                


            # print data
            if self.args.print_every > 0 and (step % self.args.print_every == 0):
                info['src'] = self.fields[dataflow[0]].reverse(source_outputs)
                info['trg'] = self.fields[dataflow[1]].reverse(target_outputs)

                if targets[1].dim() == 2:
                    info['reorder'] = self.fields[dataflow[1]].reverse(targets[1])
                else:
                    info['reorder'] = self.fields[dataflow[1]].reverse(targets[1][:, 0])

        elif mode == 'train':
            
            info['sents']  = (target_inputs[:, 0] * 0 + 1).sum()
            info['tokens'] = (target_masks != 0).sum()
            info['max_att'] = info['sents'] * max((source_inputs[0, :] * 0 + 1).sum() ** 2,
                                                  (target_inputs[0, :] * 0 + 1).sum() ** 2)

            if target_inputs.dim() == 3:  # replicate 
                W = target_inputs.size(1)
                outs = target_inputs.float() * target_masks

                # expanding source information
                for i in range(len(encoding_outputs)):
                    encoding_outputs[i] = self.expand_stack(encoding_outputs[i], W)
                source_masks = self.expand_stack(source_masks, W)

                target_inputs, target_outputs, target_masks, target_positions = \
                self.expand_stack(target_inputs), self.expand_stack(target_outputs), \
                self.expand_stack(target_masks), self.expand_stack(target_positions)

                if weights is not None:
                    info['L@D'] = ((abs(outs[:, :, None] - outs[:, None, :]).sum(-1) == 0).float().cumsum(2).cumsum(2) == 1).float().max(1)[0].mean() * W
                    info['L@W'] = ((weights ** 2).sum(-1, keepdim=True) ** (-1)).mean() # effective sample size
                    weights = self.expand_stack(weights) * W

            sources = [encoding_outputs, source_masks]
            targets = [target_inputs, target_outputs, target_masks, target_positions]
            loss = self.supervised_training(sources, targets, weights)

            # sum-up all losses
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

            
            # visualize attention
            if self.attention_flag:
                self.plot_attention(source_inputs, targets[0], dataflow)

        elif mode == 'search_train':
            
            assert self.args.path_temp == 0, 'only works for top1 search (faster)'

            info['sents']  = (target_inputs[:, 0] * 0 + 1).sum()
            info['tokens'] = (target_masks != 0).sum()
            info['max_att'] = info['sents'] * max((source_inputs[0, :] * 0 + 1).sum() ** 2,
                                                  (target_inputs[0, :] * 0 + 1).sum() ** 2)
            # info['L@G'] = min(self.args.gamma, (step - self.args.esteps) / (self.args.gsteps))

            with torch.no_grad():
                sources = [encoding_outputs, source_masks]
                targets = [target_inputs, target_outputs, target_masks, target_positions]
                targets, _ = self.search_optimal_order(sources, targets, self.args.beta, gamma=info['L@G'])
            
            loss = self.supervised_training(sources, targets)

            # sum-up all losses
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

            
            # visualize attention
            if self.attention_flag:
                self.plot_attention(source_inputs, targets[0], dataflow)

            # print data
            if self.args.print_every > 0 and (step % self.args.print_every == 0):
                info['src'] = self.fields[dataflow[0]].reverse(source_outputs)
                info['trg'] = self.fields[dataflow[1]].reverse(target_outputs)

                if targets[1].dim() == 2:
                    info['reorder'] = self.fields[dataflow[1]].reverse(targets[1])
                else:
                    info['reorder'] = self.fields[dataflow[1]].reverse(targets[1][:, 0])


        elif mode == 'decoding':

            info['sents']  = (target_inputs[:, 0] * 0 + 1).sum()
            info['tokens'] = (target_masks != 0).sum()
            info['max_att'] = info['sents'] * max((source_inputs[0, :] * 0 + 1).sum() ** 2,
                                                  (target_inputs[0, :] * 0 + 1).sum() ** 2)

            if self.args.beam_size == 1:
                translation_outputs, translation_pointers, translation_directions = self.greedy_decoding(encoding_outputs, source_masks, field=dataflow[1])
            else:
                # translation_outputs, translation_pointers, translation_directions = self.enhanced_beam_search(encoding_outputs, source_masks, field=dataflow[1], 
                #                                                                                     width=self.args.beam_size, alpha=self.args.alpha)
                translation_outputs, translation_pointers, translation_directions = self.beam_search(encoding_outputs, source_masks, field=dataflow[1], 
                                                                                                    width=self.args.beam_size, alpha=self.args.alpha)

            info['src'] = self.fields[dataflow[0]].reverse(source_outputs)
            info['trg'] = self.fields[dataflow[1]].reverse(target_outputs)
            reversed_outputs, original_outputs = self.reverse(translation_outputs, translation_pointers, translation_directions, dataflow[1])
            info['dec'], info['ori'] = reversed_outputs, original_outputs

        return info
        
    def supervised_training(self, sources, targets, weights=None):
        encoding_outputs, source_masks = sources
        target_inputs, target_outputs, target_masks, target_positions = targets
        target_len = target_inputs.size(1)
        d_model = encoding_outputs[0].size(2)

        decoding_inputs = self.io_dec.i(target_inputs, pos=False)   
        decoding_relation = (target_positions[:, :-1, None] - target_positions[:, None, :-1]).clamp(-1, 1)
        decoding_pointers = (target_positions[:, :, None] - target_positions[:, None, :]) \
                            * target_outputs.new_ones(target_len + 1, target_len + 1).tril(-1)[None, :, :]
        decoding_pointers = decoding_pointers[:, 2:]

        insertion_r = (decoding_pointers + BIG * (decoding_pointers <= 0).long()).min(-1)[1]
        insertion_l = (decoding_pointers - BIG * (decoding_pointers >= 0).long()).max(-1)[1]

        # get the hidden state representions
        decoding_outputs = self.decoder(decoding_inputs, target_masks, encoding_outputs, source_masks, decoding_relation)
        pointers  = self.pos_transform(decoding_outputs[-1])
        pointer_key_l, pointer_key_r, pointer_query, content_prediction = \
        pointers[:, :, :d_model], pointers[:, :, d_model:d_model*2], pointers[:, :, d_model*2:d_model*3], pointers[:, :, d_model*3:]

        # get loss
        if self.insert_mode == 'position_first':
            loss_pointer, pointer_out = self.pointing(pointer_query, pointer_key_l, pointer_key_r, target_masks, insertion_l, insertion_r, weights=weights)
            loss = self.io_dec.cost(target_outputs[:, 1:], target_masks[:, 1:], outputs=content_prediction[:, 1:] + pointer_out, label_smooth=self.args.label_smooth, weights=weights)

        elif self.insert_mode == 'word_first':
            loss = self.io_dec.cost(target_outputs[:, 1:], target_masks[:, 1:], outputs=content_prediction[:, 1:], label_smooth=self.args.label_smooth, weights=weights)
            target_embed = self.io_dec.i(target_outputs, pos=False)
            loss_pointer = self.pointing(self.fusion(pointer_query, target_embed), pointer_key_l, pointer_key_r, target_masks, insertion_l, insertion_r, output=False, weights=weights)

        elif self.insert_mode == 'balanced':  # gating: balance word first and position first.
            raise NotImplementedError
        else:
            raise NotImplementedError

        loss['POINTER'] = loss_pointer # * 0.2    
        return loss

    def randomize_order(self, targets):
        target_inputs, target_outputs, target_masks, target_positions = targets

        # -- uniformly generate scores in a fixed size (in order to keep sync with other processes)  # BE CAREFUL!!
        scores = target_masks.new_zeros(self.args.batch_size * (self.args.world_size * 2)).uniform_()  # randomized order.
        scores = scores[self.args.batch_size * self.args.local_rank * 2: self.args.batch_size * self.args.local_rank * 2 + target_positions.numel()].view(target_positions.size())

        scores[:, 0] = -2  # <init>
        scores[:, 1] = -1  # <stop>
        eos_mask = target_masks.clone()
        eos_mask[:, :-1] = target_masks[:, :-1] - target_masks[:, 1:]  # <eos>
        
        scores[:, 1:] = scores[:, 1:] + eos_mask * BIG
        scores[:, 1:] = scores[:, 1:] + (1 - target_masks) * (2 * BIG)

        rand_orders = scores.sort(1)[1]  # B x T + 1
        target_positions = target_positions.gather(1, rand_orders)
        target_outputs = target_outputs.gather(1, rand_orders[:, 1:] - 1)
        target_inputs[:, 1:] = target_outputs[:, :-1]

        return [target_inputs, target_outputs, target_masks, target_positions]

    def search_optimal_order(self, sources, targets, width=2, field='trg', gamma=1, epsilon=0): 

        encoding_outputs, source_masks = sources
        target_inputs, target_outputs, target_masks, target_positions = targets
        encoding = self.decoder.prepare_encoder(encoding_outputs)

        B, Ts, C = encoding[0].size()
        T = target_inputs.size(1)
        W = min(width, T - 1)

        # expanding source information
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(B, W, Ts, C).contiguous().view(B * W, Ts, C)
        mask_src = source_masks[:, None, :].expand(B, W, Ts).contiguous().view(B * W, Ts)

        # searched output sequences/positions.
        outs = torch.cat([target_inputs[:, :1], target_outputs], 1)[:, None, :].expand(B, W, T+1).contiguous()
        poss = target_positions[:, None, :].expand(B, W, T+1).contiguous()

        # candidate sequences.
        cand_pos = target_positions[:, None, 2:].expand(B, W, T-1).contiguous()
        cand_trg = target_outputs[:, None, 1:].expand(B, W, T-1).contiguous()
        mask_trg = target_masks[:, None, 1:].expand(B, W, T-1).contiguous()
        mask_trg[:, :, :-1] = mask_trg[:, :, 1:]  
        mask_trg[:, :, -1] = 0     # <eos>  cannot be selected 

        # cummulative likelihood.
        logps = encoding[0].new_zeros(B, W).float() # cumulative scores for log P(Y, \pi | X)
        logqs = encoding[0].new_zeros(B, W).float() # cumulative scores for log Q(\pi | X, Y)
        hiddens = [encoding[0].new_zeros(B, W, T, C) for l in range(len(self.decoder.layers) + 1)]

        eos_yet = encoding[0].new_zeros(B, W).byte() # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(INF)[:, :, None].expand(B, W, T).contiguous()  # --- BUG, logps < 0 assign INF here 
                                                                                        # --- UPDATE: Aug 9, 2018: BUG again, expand needs contiguous
                                                                                        # --- otherwise everything will become 0.
        eos_mask[:, :, 0] = 0  # batch x beam x T - t

        # initialization the relation matrix
        relation = outs.new_zeros(B, W, T + 1, T + 1).long()
        relation[:, :, 0] = -1
        relation[:, :, 1] = 1
        relation[:, :, 0, 0] = relation[:, :, 1, 1] = 0

        # pointers
        pointer_keys = [encoding[0].new_zeros(B, W, T, C) for l in range(2)]  # left-key / right-key
        pointer_pos = encoding[0].new_zeros(B, W, T).long()
        pointer_dir = encoding[0].new_zeros(B, W, T).long()


        for t in range(T - 1):
            hiddens[0][:, :, t] = self.decoder.prepare_embedding(hiddens[0][:, :, t] + self.io_dec.i(outs[:, :, t], pos=False))

            # encoding the parial sentences
            relation_ = relation.view(B * W, T + 1, T + 1)

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.decoder.layers[l].selfattn(x[:, -1:, :], x, x, None, relation_[:, t:t+1, :t+1])
                hiddens[l + 1][:, :, t] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l], encoding[l], mask_src)).view(B, W, C)

            pointers = self.pos_transform(hiddens[-1][:, :, t:t+1])
            pointer_keys[0][:, :, t:t+1] = pointers[:, :, :, :C]
            pointer_keys[1][:, :, t:t+1] = pointers[:, :, :, C:C*2]
            pointer_query = pointers[:, :, :, C*2:C*3]   # B x W x 1 x C
            predictor     = pointers[:, :, 0, C*3:]

            if t == 0:
                continue

            # predict word.
            word_logps = log_softmax(self.io_dec.o(predictor))
            word_logps = word_logps.gather(2, cand_trg)  # B x W x T-1

            # predict position.
            query, key1, key2 = self.fusion(pointer_query.view(B*W, -1, C), self.io_dec.i(cand_trg, pos=False).view(B*W, -1, C)), \
                                pointer_keys[0][:, :, :t+1].view(B*W, -1, C), pointer_keys[1][:, :, :t+1].view(B*W, -1, C)
            pointer_scores = self.pointing(query, key1, key2, output=False, non_causal=True).squeeze(1)  
            pointer_poses = cand_pos[:, :, :, None] - poss[:, :, None, :t+1] # B x W x T-t x (t+1)
            pointer_ins_r = (pointer_poses + BIG * (pointer_poses <= 0).long()).min(-1, keepdim=True)[1]
            pointer_ins_l = (pointer_poses - BIG * (pointer_poses >= 0).long()).max(-1, keepdim=True)[1]
            
            # pointer_logps = log_softmax(pointer_scores).view(B, W, T-t, 2*t+2)
            # pointer_logps = torch.max(pointer_logps[:, :, :, :t+1].gather(3, pointer_ins_l), pointer_logps[:, :, :, t+1:].gather(3, pointer_ins_r)).squeeze(-1)  # B x W x T-t

            pointer_probs = softmax(pointer_scores).view(B, W, T-t, 2*t+2)
            pointer_logps = torch.log(pointer_probs[:, :, :, :t+1].gather(3, pointer_ins_l) + pointer_probs[:, :, :, t+1:].gather(3, pointer_ins_r) + TINY).squeeze(-1)  # B x W x T-t

            # get together and select top K words/positons
            topk2_logps = (word_logps * gamma + pointer_logps + (1 - gamma)) * mask_trg - INF * (1 - mask_trg)  # B x W x T
            topk2_logps = topk2_logps * (eos_yet[:, :, None].float() * eos_mask[:, :, :-t] + 1 - eos_yet[:, :, None].float())
            topk2_logqs = log_softmax(topk2_logps)  # B x W x T-t
            topk2_logqs = topk2_logqs * (eos_yet[:, :, None].float() * eos_mask[:, :, :-t] + 1 - eos_yet[:, :, None].float())
            
            # mask out the sentences which are finished
            topk2_logps = topk2_logps + logps[:, :, None]
            log_scores  = topk2_logps

            if self.args.use_gumbel:
                log_scores = log_scores + log_scores.new_zeros(log_scores.size()).uniform_().add_(TINY).log_().neg_().add_(TINY).log_().neg_()

            if t == 1:  # <stop>: only choose the first beam.
                log_scores, topk_inds = log_scores[:, 0].topk(W, dim=-1)
            else:
                log_scores, topk_inds = log_scores.view(B, W * (T-t)).topk(W, dim=-1)

            topk_beam_inds = topk_inds // (T - t)  # which beam is selected.
            topk_cand_inds = topk_inds  % (T - t)  # which word is selected

            logps = topk2_logps.view(B, W * (T-t)).gather(1, topk_inds)
            logqs = logqs.gather(1, topk_beam_inds) + topk2_logqs.view(B, W * (T-t)).gather(1, topk_inds)

            # gather eos_yet
            eos_yet = eos_yet.gather(1, topk_beam_inds)

            # gather pos and target
            cand_pos = cand_pos.gather(1, topk_beam_inds[:, :, None].expand_as(cand_pos))
            cand_trg = cand_trg.gather(1, topk_beam_inds[:, :, None].expand_as(cand_trg))
            selected_pos = cand_pos.gather(2, topk_cand_inds[:, :, None]).squeeze(-1)
            selected_trg = cand_trg.gather(2, topk_cand_inds[:, :, None]).squeeze(-1)

            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs))
            poss = poss.gather(1, topk_beam_inds[:, :, None].expand_as(poss))
            outs[:, :, t+1] = selected_trg
            poss[:, :, t+1] = selected_pos
            relation[:, :, :t+2, :t+2] = (poss[:, :, :t+2, None] - poss[:, :, None, :t+2]).clamp(-1, 1)

            # gather hidden states
            topk_beam_inds = topk_beam_inds[:, :, None, None].expand_as(hiddens[0]).contiguous()
            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds)

            # gather pointer keys (be careful, they are already reordered once!!)
            pointer_keys[0] = pointer_keys[0].gather(1, topk_beam_inds)
            pointer_keys[1] = pointer_keys[1].gather(1, topk_beam_inds)

            # gather and update candidates
            unselected = (1 - cand_pos.new_zeros(B, W, T-t).scatter_(2, topk_cand_inds[:, :, None], 1)).byte()
            cand_pos = cand_pos[unselected].view(B, W, T-t-1)
            cand_trg = cand_trg[unselected].view(B, W, T-t-1)
            mask_trg = mask_trg[:, :, 1:]
            
            eos_yet  = eos_yet | (mask_trg.sum(2) == 0)

        # output data
        if self.args.path_temp == 0:  # pick the top path
            target_inputs  = outs[:, 0, :-1].contiguous()
            target_outputs = outs[:, 0,  1:].contiguous()
            target_positions = poss[:, 0]
            weights = None

        else:    

            target_inputs  = outs[:, :, :-1].contiguous()
            target_outputs = outs[:, :,  1:].contiguous()
            target_positions = poss
            target_masks = target_masks[:, None, :].expand(B, W, T).contiguous()

            if not self.args.no_weights:
                weights = softmax((logps - logqs) / self.args.path_temp).detach()
            else:
                weights = None

        targets = [target_inputs, target_outputs, target_masks, target_positions]
        return targets, weights
        
    def greedy_decoding(self, encoding=None, mask_src=None, T=None, field='trg'):

        encoding = self.decoder.prepare_encoder(encoding)
        if T is None:
            T = encoding[0].size()[1]
        B, C = encoding[0].size()[0], encoding[0].size()[-1]  # batch_size, decoding-length, size
        T *= self.length_ratio

        outs = encoding[0].new_zeros(B, T + 2).long().fill_(self.fields[field].vocab.stoi[self.fields[field].init_token])
        outs[:, 1] = self.fields[field].vocab.stoi['<stop>']

        hiddens = [encoding[0].new_zeros(B, T + 1, C) for l in range(len(self.decoder.layers) + 1)]
        eos_yet = encoding[0].new_zeros(B).byte()

        # initialization the relation matrix
        relation = outs.new_zeros(B, T + 1, T + 1).long()
        relation[:, 0] = -1
        relation[:, 1] = 1
        relation[:, 0, 0] = relation[:, 1, 1] = 0

        # pointers
        pointer_keys = [encoding[0].new_zeros(B, T + 1, C) for l in range(2)]  # left-key / right-key
        pointer_pos = encoding[0].new_zeros(B, T + 1).long()
        pointer_dir = encoding[0].new_zeros(B, T + 1).long()


        for t in range(T):
            
            # add dropout, etc.
            hiddens[0][:, t] = self.decoder.prepare_embedding(hiddens[0][:, t] + self.io_dec.i(outs[:, t], pos=False))

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :t+1]
                x = self.decoder.layers[l].selfattn(hiddens[l][:, t:t+1], x, x, None, relation[:, t:t+1, :t+1])   # we need to make the dimension 3D
                hiddens[l + 1][:, t] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l], encoding[l], mask_src))[:, 0]

            pointers = self.pos_transform(hiddens[-1][:, t:t+1])
            pointer_keys[0][:, t:t+1] = pointers[:, :, :C]
            pointer_keys[1][:, t:t+1] = pointers[:, :, C:C*2]
            pointer_query = pointers[:, :, C*2:C*3]
            predictor = pointers[:, 0, C*3:]

            if t == 0: # do not need to predict, next word is <stop> 
                continue
            
            else: 
                
                # predict the insertion location jointly with the word
                if self.insert_mode == 'position_first':
                    pointers, pointer_out = self.pointing(pointer_query, pointer_keys[0][:, :t+1], pointer_keys[1][:, :t+1])   # batch-size
                    _, preds = self.io_dec.o((predictor + pointer_out) * 0.5 ** 0.5).max(-1)

                elif self.insert_mode == 'word_first':
                    _, preds = self.io_dec.o(predictor).max(-1)
                    pointers = self.pointing(self.fusion(pointer_query, self.io_dec.i(preds[:, None], pos=False)), pointer_keys[0][:, :t+1], pointer_keys[1][:, :t+1], output=False)
                    pointers = pointers.squeeze(1).max(1)[1]
                else:
                    raise NotImplementedError

                preds[eos_yet] = self.fields[field].vocab.stoi['<pad>']
                eos_yet = eos_yet | (preds == self.fields[field].vocab.stoi['<eos>'])
                outs[:, t + 1] = preds
                
                pointer_dir[:, t] = pointers // (t + 1)   # left or right
                pointer_pos[:, t] = pointers % (t + 1)    # pointing to which word to insert

                # insert the token and update the relation
                new_relation = relation[:, :t+2, :t+2].gather(1, pointer_pos[:, t:t+1, None].expand(B, 1, t+2)).squeeze(1)
                new_relation.scatter_(1, pointer_pos[:, t:t+1], 2 * pointer_dir[:, t:t+1] - 1)
                new_relation[:, t+1] = 0

                relation[:, t+1, :t+2] = new_relation
                relation[:, :t+2, t+1] = -new_relation

                # early stop
                if eos_yet.all():
                    break
                
        return outs[:, 2:t+2], pointer_pos[:, 1:t+1], pointer_dir[:, 1:t+1]

    def beam_search(self, encoding=None, mask_src=None, width=2, alpha=0.6, T=None, field='trg'):  # width: beamsize, alpha: length-norm
        assert self.insert_mode == 'word_first', 'currently word-first is in default.'

        encoding = self.decoder.prepare_encoder(encoding)

        W = width
        if T is None:
            T = encoding[0].size()[1]
        B, C = encoding[0].size()[0], encoding[0].size()[-1]  # batch_size, decoding-length, size

        # expanding
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(B, W, T, C).contiguous().view(B * W, T, C)
        mask_src = mask_src[:, None, :].expand(B, W, T).contiguous().view(B * W, T)

        T *= self.length_ratio
        outs = encoding[0].new_zeros(B, W, T + 1).long().fill_(self.fields[field].vocab.stoi['<pad>'])
        outs[:, :, 0] = self.fields[field].vocab.stoi[self.fields[field].init_token]
        outs[:, :, 1] = self.fields[field].vocab.stoi['<stop>']

        logps = encoding[0].new_zeros(B, W).float() # scores
        hiddens = [encoding[0].new_zeros(B, W, T, C) for l in range(len(self.decoder.layers) + 1)]

        eos_yet = encoding[0].new_zeros(B, W).byte() # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(INF)[:, :, None].expand(B, W, W).contiguous()  # --- BUG, logps < 0 assign INF here 
                                                                                        # --- UPDATE: Aug 9, 2018: BUG again, expand needs contiguous
                                                                                        # --- otherwise everything will become 0.
        eos_mask[:, :, 0] = 0  # batch x beam x beam


        # initialization the relation matrix
        relation = outs.new_zeros(B, W, T + 1, T + 1).long()
        relation[:, :, 0] = -1
        relation[:, :, 1] = 1
        relation[:, :, 0, 0] = relation[:, :, 1, 1] = 0

        # pointers
        pointer_keys = [encoding[0].new_zeros(B, W, T, C) for l in range(2)]  # left-key / right-key
        pointer_pos = encoding[0].new_zeros(B, W, T).long()
        pointer_dir = encoding[0].new_zeros(B, W, T).long()

        for t in range(T):
            hiddens[0][:, :, t] = self.decoder.prepare_embedding(hiddens[0][:, :, t] + self.io_dec.i(outs[:, :, t], pos=False))

            # encoding the parial sentences
            relation_ = relation.view(B * W, T + 1, T + 1)

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.decoder.layers[l].selfattn(x[:, -1:, :], x, x, None, relation_[:, t:t+1, :t+1])
                hiddens[l + 1][:, :, t] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l], encoding[l], mask_src)).view(B, W, C)

            pointers = self.pos_transform(hiddens[-1][:, :, t:t+1])
            pointer_keys[0][:, :, t:t+1] = pointers[:, :, :, :C]
            pointer_keys[1][:, :, t:t+1] = pointers[:, :, :, C:C*2]
            pointer_query = pointers[:, :, :, C*2:C*3]
            predictor     = pointers[:, :, 0, C*3:]

            if t == 0:
                continue

            # predict word-first.
            # topk2_logps: scores, topk2_inds: top word index at each beam, batch x beam x beam
            topk2_logps = log_softmax(self.io_dec.o(predictor))
            topk2_logps[:, :, self.fields[field].vocab.stoi['<pad>']] = -INF
            topk2_logps, topk2_inds = topk2_logps.topk(W, dim=-1)

            # mask out the sentences which are finished
            topk2_logps = topk2_logps * (eos_yet[:, :, None].float() * eos_mask + 1 - eos_yet[:, :, None].float())
            topk2_logps = topk2_logps + logps[:, :, None]

            if t == 1:  # <stop>: only choose the first beam.
                logps, topk_inds = topk2_logps[:, 0].topk(W, dim=-1)
            else:
                logps, topk_inds = topk2_logps.view(B, W * W).topk(W, dim=-1)

            topk_beam_inds  = topk_inds.div(W)  # which beam is selected.
            topk_token_inds = topk2_inds.view(B, W * W).gather(1, topk_inds)  # W words selected

            # gather pointer keys & query
            pointer_keys[0] = pointer_keys[0].gather(1, topk_beam_inds[:, :, None, None].expand(B, W, T, C)) # reorder pointer keys/query
            pointer_keys[1] = pointer_keys[1].gather(1, topk_beam_inds[:, :, None, None].expand(B, W, T, C))
            pointer_query = pointer_query.gather(1, topk_beam_inds[:, :, None, None].expand(B, W, 1, C))

            # pointing to a location
            query, key1, key2 = self.fusion(pointer_query.view(B*W, -1, C), self.io_dec.i(topk_token_inds[:, :, None], pos=False).view(B*W, -1, C)), \
                                pointer_keys[0][:, :, :t+1].view(B*W, -1, C), pointer_keys[1][:, :, :t+1].view(B*W, -1, C)
            pointer_scores = self.pointing(query, key1, key2, output=False, non_causal=True).squeeze(1)  
            # query, key1, key2 = self.fusion(pointer_query, self.io_dec.i(topk_token_inds[:, :, None], pos=False)), pointer_keys[0][:, :, :t+1], pointer_keys[1][:, :, :t+1]
            # pointer_scores = self.pointing(query.view(B*W, -1, C), key1.view(B*W, -1, C), key2.view(B*W, -1, C), output=False).squeeze(1)  # B x W x (2t+2)
            pointer_logps  = log_softmax(pointer_scores).view(B, W, -1) + logps[:, :, None] # B x -1

            logps, topk_pointer_inds = pointer_logps.view(B, -1).topk(W, dim=-1) # beam-search again for top W words/positions.
            topk_beam_inds0 = topk_pointer_inds // (2 * t + 2)
            topk_pos_inds   = topk_pointer_inds %  (2 * t + 2)
            topk_token_inds = topk_token_inds.gather(1, topk_beam_inds0)  # reorder the selected words
            topk_beam_inds  = topk_beam_inds.gather(1, topk_beam_inds0)   # reorder the ordered beam index

            # check if sentence ends.
            eos_yet = eos_yet.gather(1, topk_beam_inds.data)

            # logps = logps.gather(1, topk_beam_inds0)
            # gather relative position maps
            pointer_dir = pointer_dir.gather(1, topk_beam_inds[:, :, None].expand_as(pointer_dir)).contiguous()
            pointer_pos = pointer_pos.gather(1, topk_beam_inds[:, :, None].expand_as(pointer_pos)).contiguous()
            pointer_dir[:, :, t] = topk_pos_inds // (t + 1) # left/right
            pointer_pos[:, :, t] = topk_pos_inds %  (t + 1)

            # gather new relations
            relation = relation.gather(1, topk_beam_inds[:, :, None, None].expand_as(relation)).contiguous()
            new_relation = relation[:, :, :t+2, :t+2].gather(2, pointer_pos[:, :, t:t+1, None].expand(B, W, 1, t+2)).squeeze(2)

            new_relation.scatter_(2, pointer_pos[:, :, t:t+1], 2 * pointer_dir[:, :, t:t+1] - 1)
            new_relation[:, :, t+1]   = 0
            relation[:, :, t+1, :t+2] = new_relation
            relation[:, :, :t+2, t+1] = -new_relation


            # gather outputs
            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs)).contiguous()
            outs[:, :, t + 1] = topk_token_inds

            # gather hidden states
            topk_beam_inds = topk_beam_inds[:, :, None, None].expand_as(hiddens[0]).contiguous()
            topk_beam_inds0 = topk_beam_inds0[:, :, None, None].expand_as(hiddens[0]).contiguous()
            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds)

            # gather pointer keys (be careful, they are already reordered once!!)
            pointer_keys[0] = pointer_keys[0].gather(1, topk_beam_inds0)
            pointer_keys[1] = pointer_keys[1].gather(1, topk_beam_inds0)

            # alpha re-weight
            logps = logps * (1 + (eos_yet.float()) * 1 / (t + 1)).pow(alpha)
            eos_yet = eos_yet | (topk_token_inds == self.fields[field].vocab.stoi['<eos>'])

            if eos_yet.all():
                break

        return outs[:, 0, 2:t+2], pointer_pos[:, 0, 1:t+1], pointer_dir[:, 0, 1:t+1]
    
    def particle_filtering(self, sources, targets, width=2, field='trg', resampling=False, gamma=1): 

        encoding_outputs, source_masks = sources
        target_inputs, target_outputs, target_masks, target_positions = targets
        encoding = self.decoder.prepare_encoder(encoding_outputs)

        B, Ts, C = encoding[0].size()
        T = target_inputs.size(1)
        W = min(width, T - 1)

        # expanding source information
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(B, W, Ts, C).contiguous().view(B * W, Ts, C)
        mask_src = source_masks[:, None, :].expand(B, W, Ts).contiguous().view(B * W, Ts)

        # searched output sequences/positions.
        outs = torch.cat([target_inputs[:, :1], target_outputs], 1)[:, None, :].expand(B, W, T+1).contiguous()
        poss = target_positions[:, None, :].expand(B, W, T+1).contiguous()

        # candidate sequences.
        cand_pos = target_positions[:, None, 2:].expand(B, W, T-1).contiguous()
        cand_trg = target_outputs[:, None, 1:].expand(B, W, T-1).contiguous()
        mask_trg = target_masks[:, None, 1:].expand(B, W, T-1).contiguous()
        mask_trg[:, :, :-1] = mask_trg[:, :, 1:]  
        mask_trg[:, :, -1] = 0     # <eos>  cannot be selected 

        # cummulative likelihood.
        logps = encoding[0].new_zeros(B, W).float() # cumulative scores for log P(Y, \pi | X)
        logqs = encoding[0].new_zeros(B, W).float() # cumulative scores for log Q(\pi | X, Y)
        weights = encoding[0].new_ones(B, W).float() / W  # initially uniform

        hiddens = [encoding[0].new_zeros(B, W, T, C) for l in range(len(self.decoder.layers) + 1)]

        eos_yet = encoding[0].new_zeros(B, W).byte() # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(INF)[:, :, None].expand(B, W, T).contiguous()  # --- BUG, logps < 0 assign INF here 
                                                                                        # --- UPDATE: Aug 9, 2018: BUG again, expand needs contiguous
                                                                                        # --- otherwise everything will become 0.
        eos_mask[:, :, 0] = 0  # batch x beam x T - t

        # initialization the relation matrix
        relation = outs.new_zeros(B, W, T + 1, T + 1).long()
        relation[:, :, 0] = -1
        relation[:, :, 1] = 1
        relation[:, :, 0, 0] = relation[:, :, 1, 1] = 0

        # pointers
        pointer_keys = [encoding[0].new_zeros(B, W, T, C) for l in range(2)]  # left-key / right-key
        pointer_pos = encoding[0].new_zeros(B, W, T).long()
        pointer_dir = encoding[0].new_zeros(B, W, T).long()

        # used in resampling
        normal_beam_inds = torch.arange(W, device=outs.get_device()).long()[None, :].expand(B, W)
        normal_weights = weights * 0 + 1 / W
        for t in range(T - 1):
            hiddens[0][:, :, t] = self.decoder.prepare_embedding(hiddens[0][:, :, t] + self.io_dec.i(outs[:, :, t], pos=False))

            # encoding the parial sentences
            relation_ = relation.view(B * W, T + 1, T + 1)

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.decoder.layers[l].selfattn(x[:, -1:, :], x, x, None, relation_[:, t:t+1, :t+1])
                hiddens[l + 1][:, :, t] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l], encoding[l], mask_src)).view(B, W, C)

            pointers = self.pos_transform(hiddens[-1][:, :, t:t+1])
            pointer_keys[0][:, :, t:t+1] = pointers[:, :, :, :C]
            pointer_keys[1][:, :, t:t+1] = pointers[:, :, :, C:C*2]
            pointer_query = pointers[:, :, :, C*2:C*3]   # B x W x 1 x C
            predictor     = pointers[:, :, 0, C*3:]

            if t == 0:
                continue

            # predict word.
            word_logps = log_softmax(self.io_dec.o(predictor))
            word_logps = word_logps.gather(2, cand_trg)  # B x W x T-1

            # predict position.
            query, key1, key2 = self.fusion(pointer_query.view(B*W, -1, C), self.io_dec.i(cand_trg, pos=False).view(B*W, -1, C)), \
                                pointer_keys[0][:, :, :t+1].view(B*W, -1, C), pointer_keys[1][:, :, :t+1].view(B*W, -1, C)
            pointer_scores = self.pointing(query, key1, key2, output=False, non_causal=True).squeeze(1)  
            
            pointer_poses = cand_pos[:, :, :, None] - poss[:, :, None, :t+1] # B x W x T-t x (t+1)
            pointer_ins_r = (pointer_poses + BIG * (pointer_poses <= 0).long()).min(-1, keepdim=True)[1]
            pointer_ins_l = (pointer_poses - BIG * (pointer_poses >= 0).long()).max(-1, keepdim=True)[1]
            
            # pointer_logps = log_softmax(pointer_scores).view(B, W, T-t, 2*t+2)
            # pointer_logps = torch.max(pointer_logps[:, :, :, :t+1].gather(3, pointer_ins_l), pointer_logps[:, :, :, t+1:].gather(3, pointer_ins_r)).squeeze(-1)  # B x W x T-t

            pointer_probs = softmax(pointer_scores).view(B, W, T-t, 2*t+2)
            pointer_logps = torch.log(pointer_probs[:, :, :, :t+1].gather(3, pointer_ins_l) + pointer_probs[:, :, :, t+1:].gather(3, pointer_ins_r) + TINY).squeeze(-1)  # B x W x T-t

            # get together and select top K words/positons
            topk2_logps = (word_logps + pointer_logps) * mask_trg / gamma - INF * (1 - mask_trg)  # B x W x T
            topk2_logps = topk2_logps * (eos_yet[:, :, None].float() * eos_mask[:, :, :-t] + 1 - eos_yet[:, :, None].float())
            # topk2_logqs = log_softmax(topk2_logps)  # B x W x T-t
            # topk2_logqs = topk2_logqs * (eos_yet[:, :, None].float() * eos_mask[:, :, :-t] + 1 - eos_yet[:, :, None].float())
            
            # :: sample at each beam saperately::
            topk2_qs = softmax(topk2_logps) # B x W x T-t
            weights1  = weights * torch.sum(torch.exp(topk2_logps), 2)
            weights1  = weights1 / weights1.sum(1, keepdim=True)
            
            # refining the weights (to avoid the path/particle degeneracy)
            weights0 = weights * 0 + 1 / W
            weights0 = weights0 * ((abs(outs[:, :, None, :t+1] - outs[:, None, :, :t+1]).sum(-1) == 0).float().cumsum(2).cumsum(2) == 1).float().sum(1)
            weights2 = weights1 * weights0 * W  # make sure to collpase the equvalient beams --- resampling weights ----

            topk_inds = stratified_resample((topk2_qs * weights0[:,:,None]).view(B, -1), W)
            topk_beam_inds0 = topk_inds // (T-t)
            topk_cand_inds0 = topk_inds % (T-t)
            
            topk_inds = stratified_resample((topk2_qs * weights2[:,:,None]).contiguous().view(B, -1) , W)
            topk_beam_inds1 = topk_inds // (T-t)
            topk_cand_inds1 = topk_inds % (T-t)

            # effective particles
            effective_sample_size = ((weights1 ** 2).sum(-1, keepdim=True) ** (-1))
            resampling_choice = (effective_sample_size < (self.args.adaptive_ess_ratio * W))

            # resampling when it is needed
            # never resampling after the beam ends
            weights = (resampling_choice.float() * normal_weights + (1 - resampling_choice.float()) * weights1) * (1 - eos_yet.float()) + weights * eos_yet.float()
            topk_beam_inds = resampling_choice.long() * topk_beam_inds1 + (1 - resampling_choice.long()) * topk_beam_inds0  
            topk_beam_inds = topk_beam_inds * (1 - eos_yet.long()) + normal_beam_inds * eos_yet.long()
            topk_cand_inds = resampling_choice.long() * topk_cand_inds1 + (1 - resampling_choice.long()) * topk_cand_inds0 
            eos_yet = eos_yet.gather(1, topk_beam_inds)

            # gather pos and target
            cand_pos = cand_pos.gather(1, topk_beam_inds[:, :, None].expand_as(cand_pos))
            cand_trg = cand_trg.gather(1, topk_beam_inds[:, :, None].expand_as(cand_trg))
            selected_pos = cand_pos.gather(2, topk_cand_inds[:, :, None]).squeeze(-1)
            selected_trg = cand_trg.gather(2, topk_cand_inds[:, :, None]).squeeze(-1)

            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs))
            poss = poss.gather(1, topk_beam_inds[:, :, None].expand_as(poss))
            outs[:, :, t+1] = selected_trg
            poss[:, :, t+1] = selected_pos
            relation[:, :, :t+2, :t+2] = (poss[:, :, :t+2, None] - poss[:, :, None, :t+2]).clamp(-1, 1)

            # gather hidden states
            topk_beam_inds2 = topk_beam_inds[:, :, None, None].expand_as(hiddens[0]).contiguous()
            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds2)

            # gather pointer keys (be careful, they are already reordered once!!)
            pointer_keys[0] = pointer_keys[0].gather(1, topk_beam_inds2)
            pointer_keys[1] = pointer_keys[1].gather(1, topk_beam_inds2)

            # gather and update candidates
            unselected = (1 - cand_pos.new_zeros(B, W, T-t).scatter_(2, topk_cand_inds[:, :, None], 1)).byte()
            cand_pos = cand_pos[unselected].view(B, W, T-t-1)
            cand_trg = cand_trg[unselected].view(B, W, T-t-1)
            mask_trg = mask_trg[:, :, 1:]
            eos_yet  = eos_yet | (mask_trg.sum(2) == 0)

        # output data
        target_inputs  = outs[:, :, :-1].contiguous()
        target_outputs = outs[:, :,  1:].contiguous()
        target_positions = poss
        target_masks = target_masks[:, None, :].expand(B, W, T).contiguous()

        if self.args.no_weights:
            weights = None
        
        targets = [target_inputs, target_outputs, target_masks, target_positions]
        return targets, weights
        


    