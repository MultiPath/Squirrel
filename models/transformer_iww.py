from .core import *
from .transformer import Transformer
from .transformer_ins import TransformerIns
import copy

class TransformerIww(TransformerIns):
    
    def __init__(self, src, trg, args):
        super().__init__(src, trg, args)
        self.inferencer = Stack(src, args, causal=False, cross=True)
        self.inference_attention = Attention(args.d_model, 1, 0, causal=False)

    def forward(self, batch, mode='train', reverse=True, dataflow=['src', 'trg'], step=None):
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

        # encoding
        encoding_inputs  = self.io_enc.i(source_inputs, pos=False)
        encoding_outputs = self.encoder(encoding_inputs, source_masks)

        if mode == 'train':

            info['sents']  = (target_inputs[:, 0] * 0 + 1).sum()
            info['tokens'] = (target_masks != 0).sum()
            info['max_att'] = info['sents'] * max((source_inputs[0, :] * 0 + 1).sum() ** 2,
                                                  (target_inputs[0, :] * 0 + 1).sum() ** 2)
                
            sources = [encoding_outputs, source_masks]
            targets = [target_inputs, target_outputs, target_masks, target_positions]
            loss, new_targets = self.supervised_training(sources, targets)

            if self.args.print_every > 0 and (step % self.args.print_every == 0):
                info['src'] = self.fields[dataflow[0]].reverse(source_outputs[:1])
                info['trg'] = self.fields[dataflow[1]].reverse(target_outputs[:1])
                info['reorder'] = self.fields[dataflow[1]].reverse(new_targets[0][:1])

            # sum-up all losses
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

            
            # visualize attention
            if self.attention_flag:
                self.plot_attention(source_inputs, targets[0], dataflow)

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
        
    def supervised_training(self, sources, targets):
        loss = dict()

        encoding_outputs, source_masks = sources
        target_inputs, target_outputs, target_masks, target_positions = targets
        T = target_inputs.size(1)

        decoding_inputs = self.io_dec.i(target_inputs, pos=False) 
        B, Ts, C = encoding_outputs[0].size()
        W = self.args.beta

        # ------- inference network: encode the target sentence --------- #
        decoding_inputs = self.io_dec.i(target_inputs, pos=False)   
        decoding_relation = (target_positions[:, :-1, None] - target_positions[:, None, :-1]).clamp(-1, 1)
        inference_decoding_outputs = self.inferencer(decoding_inputs, target_masks, encoding_outputs, source_masks, decoding_relation)
        inference_attention_scores = self.inference_attention(inference_decoding_outputs[-1], inference_decoding_outputs[-1], mask=target_masks)  # B x T x T

        # ------- sampling based on the inference scores ---------------- #
        # expanding information
        target_masks = target_masks[:, None, :].expand(B, W, T).contiguous().view(B * W, T)
        target_outputs = target_outputs[:, None, :].expand(B, W, T).contiguous().view(B * W, T)
        target_positions = target_positions[:, None, :].expand(B, W, T+1).contiguous().view(B * W, T+1)
        inference_attention_scores = inference_attention_scores[:, None, :, :].expand(B, W, T, T).contiguous().view(B * W, T, T)
        inference_attention_scores_w_noise = inference_attention_scores + gumbel_noise(inference_attention_scores).detach()
        inference_attention_scores_w_noise = inference_attention_scores_w_noise * target_masks[:, :, None]
        
        # linear sum assignment problem / matching problem
        selected_pos = matching(inference_attention_scores_w_noise)

        # continuous approximation of gumbel-sinkhorn
        continuous_selected_pos = gumbel_sinkhorn(inference_attention_scores, temp=1.0, n_samples=1, noise_factor=0.0)


        inference_mask = target_masks.clone()
        inference_mask[:, :2] = 0

        selected_pos  = torch.arange(T + 1, device=target_positions.get_device())[None, :].expand_as(target_positions).contiguous()
        _selected_pos = selected_pos.clone()
        likelihood_q  = []

        # start sampling step by step
        infer_sample = target_positions.new_ones(B * W)        
        for t in range(1, T-1):            
            infer_scores = inference_attention_scores.gather(1, infer_sample[:, None, None].expand(B * W, 1, T)).squeeze(1)
            infer_sample = ((infer_scores + gumbel_noise[:, t, :]) * inference_mask - INF * (1 - inference_mask)).max(1)[1]  
            likelihood_q.append(log_softmax(infer_scores * inference_mask - INF * (1 - inference_mask)).gather(1, infer_sample[:, None]))
            selected_pos[:, t+1] = infer_sample   # target_positions.gather(1, infer_sample[:, None]).squeeze(1)
            inference_mask = inference_mask.scatter(1, infer_sample[:, None], 0)

        likelihood_q = torch.cat(likelihood_q, 1)
        selected_pos[:, :-1] = selected_pos[:, :-1] * target_masks.long() + (1-target_masks.long()) * _selected_pos[:, :-1]        
        target_positions = target_positions.gather(1, selected_pos) 


        target_outputs = target_outputs.gather(1, selected_pos[:, 1:] - 1)
        decoding_inputs = decoding_inputs[:, None, :, :].expand(B, W, T, C).contiguous().view(B * W, T, C)
        decoding_inputs = decoding_inputs.gather(1, selected_pos[:, :-1, None].expand(B * W, T, C))


        # -------- Reweighted Wake-Sleep --------------------------------- #
        # use sampled positions to train translation models
        decoding_relation = (target_positions[:, :-1, None] - target_positions[:, None, :-1]).clamp(-1, 1)
        decoding_pointers = (target_positions[:, :, None] - target_positions[:, None, :]) \
                            * target_outputs.new_ones(T + 1, T + 1).tril(-1)[None, :, :]
        decoding_pointers = decoding_pointers[:, 2:]

        insertion_r = (decoding_pointers + BIG * (decoding_pointers <= 0).long()).min(-1)[1]
        insertion_l = (decoding_pointers - BIG * (decoding_pointers >= 0).long()).max(-1)[1]

        # expanding information
        for i in range(len(encoding_outputs)):
            encoding_outputs[i] = encoding_outputs[i][:, None, :].expand(B, W, Ts, C).contiguous().view(B * W, Ts, C)
        source_masks = source_masks[:, None, :].expand(B, W, Ts).contiguous().view(B * W, Ts)
        

        # get the hidden state representions

        def _forward(eval=False):
            
            if eval:
                self.eval()
            else:
                self.train()

            decoding_outputs = self.decoder(decoding_inputs, target_masks, encoding_outputs, source_masks, decoding_relation)
            pointers  = self.pos_transform(decoding_outputs[-1])
            pointer_key_l, pointer_key_r, pointer_query, content_prediction = \
            pointers[:, :, :C], pointers[:, :, C:C*2], pointers[:, :, C*2:C*3], pointers[:, :, C*3:]

            # token loss
            token_logits = log_softmax(self.io_dec.o(content_prediction[:, 1:]))
            token_smooth = token_logits.mean(-1)
            loss_token   = -token_logits.gather(2, target_outputs[:, 1:, None]).squeeze(2) * (1 - self.args.label_smooth) - token_smooth * self.args.label_smooth
            loss_token   = (loss_token * target_masks[:, 1:]).sum(-1).view(B, W)

            # position loss
            pointer_query = self.fusion(pointer_query, self.io_dec.i(target_outputs, pos=False))
            loss_pointer  = self.pointing(pointer_query, pointer_key_l, pointer_key_r, target_masks, insertion_l, insertion_r, reduction='none')
            loss_pointer  = (loss_pointer * target_masks[:, 1:]).sum(-1).view(B, W)

            # inference loss
            loss_inference = (-likelihood_q * target_masks[:, 2:]).sum(-1).view(B, W)

            return loss_token, loss_pointer, loss_inference

        # RWS: use importance sampling #
        loss_token, loss_pointer, loss_inference = _forward(True)
        importance_weights = softmax(-loss_token - loss_pointer + loss_inference).detach()
        loss_token, loss_pointer, loss_inference = _forward(False)
        
        loss_token     = (loss_token     * importance_weights).sum() / target_masks[:, 1:].sum() * W
        loss_pointer   = (loss_pointer   * importance_weights).sum() / target_masks[:, 1:].sum() * W
        loss_inference = (loss_inference * importance_weights).sum() / target_masks[:, 2:].sum() * W

        loss['#W']  = (importance_weights * torch.log(importance_weights + TINY)).sum() / target_masks[:, 1:].sum() * W
        loss['MLE'] = loss_token
        loss['POINTER'] = loss_pointer
        loss['INFER'] = loss_inference

        new_targets = [target_outputs, target_positions]

        return loss, new_targets

    def greedy_decoding(self, encoding=None, mask_src=None, T=None, field='trg'):

        encoding = self.decoder.prepare_encoder(encoding)
        if T is None:
            T = encoding[0].size()[1]
        B, C = encoding[0].size()[0], encoding[0].size()[-1]  # batch_size, decoding-length, size
        T *= self.length_ratio

        outs = encoding[0].new_zeros(B, T + 2).long().fill_(self.fields[field].vocab.stoi['<init>'])
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
                    _, preds = self.io_dec.o(predictor + pointer_out).max(-1)
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
        outs[:, :, 0] = self.fields[field].vocab.stoi['<init>']
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
            query, key1, key2 = self.fusion(pointer_query, self.io_dec.i(topk_token_inds[:, :, None], pos=False)), pointer_keys[0][:, :, :t+1], pointer_keys[1][:, :, :t+1]
            pointer_scores = self.pointing(query.view(B*W, -1, C), key1.view(B*W, -1, C), key2.view(B*W, -1, C), output=False).squeeze(1)  # B x W x (2t+2)
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
    


