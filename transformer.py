from core import *

class Transformer(nn.Module):

    def __init__(self, src, trg, args):
        super().__init__()
        self.encoder = Stack(src, args, causal=args.causal_enc, cross=False)
        self.decoder = Stack(trg, args, causal=True, cross=True)
        
        if args.multi_width > 1:
            self.io_dec = MulIO(trg, args)
            self.io_enc = IO(src, args)
        else:
            self.io_dec = IO(trg, args)
            self.io_enc = IO(src, args)

        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.length_ratio = args.length_ratio
        self.fields = {'src': src, 'trg': trg}
        self.args = args  

        # decode or not:
        self.decode = False
        
    def prepare_masks(self, inputs):
        field, text = inputs
        if text.ndimension() == 2:  # index inputs
            masks = (text.data != self.fields[field].vocab.stoi['<pad>']).float()
        else:                       # one-hot vector inputs
            masks = (text.data[:, :, self.fields[field].vocab.stoi['<pad>']] != 1).float()
        return masks

    def prepare_data(self, batch):
        source_inputs, source_outputs = batch.src[:, :-1].contiguous(), batch.src[:, 1:].contiguous()
        target_inputs, target_outputs = batch.trg[:, :-1].contiguous(), batch.trg[:, 1:].contiguous()
        source_masks, target_masks = self.prepare_masks(('src', source_outputs)), self.prepare_masks(('trg', target_outputs))
        return source_inputs, source_outputs, source_masks, target_inputs, target_outputs, target_masks


    # All in All: forward function for training
    def forward(self, batch, decoding=False, reverse=True):
        
        #if info is None:
        info = defaultdict(lambda: 0)

        source_inputs, source_outputs, source_masks, \
        target_inputs, target_outputs, target_masks = self.prepare_data(batch)

        info['sents']  = (target_inputs[:, 0] * 0 + 1).sum()
        info['tokens'] = (target_masks != 0).sum()

        # in some extreme case.
        if info['sents'] == 0:
            return info

        # encoding
        encoding_outputs = self.encoder(self.io_enc.i(source_inputs, pos=True), source_masks)
        if not decoding:
            # Maximum Likelihood Training (with label smoothing trick)

            decoding_outputs = self.decoder(self.io_dec.i(target_inputs), target_masks, encoding_outputs, source_masks)
            loss = self.io_dec.cost(target_outputs, target_masks, outputs=decoding_outputs[-1], label_smooth=self.args.label_smooth)
            
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

            # Source side Language Model (optional, only works for causal-encoder)
            if self.args.encoder_lm and self.args.causal_enc:
                loss_lm = self.io_enc.cost(source_outputs, source_masks, outputs=encoding_outputs[-1])
                for w in loss_lm:
                    info['L@' + w] = loss[w]
                    if w[0] != '#':
                        info['loss'] = info['loss'] + loss[w]

        else:
            # Decoding (for evaluation)

            if self.args.multi_width > 1: # -- the newly introduced block-wise decoding --
                assert self.args.beam_size == 1, 'block-wise decoding only works for greedy decoding (for now).' 
                translation_outputs = self.blockwise_parallel_decoding(encoding_outputs, source_masks)

            else:
                if self.args.beam_size == 1:
                    translation_outputs = self.greedy_decoding(encoding_outputs, source_masks)
                else:
                    translation_outputs = self.beam_search(encoding_outputs, source_masks, self.args.beam_size, self.args.alpha)

            if reverse:
                source_outputs = self.io_enc.reverse(source_outputs)
                target_outputs = self.io_dec.reverse(target_outputs)
                
                # specially for multi_step decoding #
                if self.args.multi_width > 1:
                    translation_outputs, saved_time, pred_acc, decisions = self.io_dec.reverse(translation_outputs, width=self.args.multi_width, return_saved_time=True)
                    
                    info['saved_time'] = saved_time
                    info['pred_acc'] = pred_acc
                    info['decisions'] = decisions
                
                else:
                    translation_outputs = self.io_dec.reverse(translation_outputs)

            info['src'] = source_outputs
            info['trg'] = target_outputs
            info['dec'] = translation_outputs
        
        return info

    def greedy_decoding(self, encoding=None, mask_src=None):

        encoding = self.decoder.prepare_encoder(encoding)
        B, T, C = encoding[0].size()  # batch_size, decoding-length, size
        T *= self.length_ratio

        outs = encoding[0].new_zeros(B, T + 1).long().fill_(self.fields['trg'].vocab.stoi['<init>'])
        hiddens = [encoding[0].new_zeros(B, T, C) for l in range(len(self.decoder.layers) + 1)]
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].new_zeros(B).byte()

        for t in range(T):
            
            # add dropout, etc.
            hiddens[0][:, t] = self.decoder.prepare_embedding(hiddens[0][:, t] + self.io_dec.i(outs[:, t], pos=False))

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :t+1]
                x = self.decoder.layers[l].selfattn(hiddens[l][:, t:t+1], x, x)   # we need to make the dimension 3D
                hiddens[l + 1][:, t] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l], encoding[l], mask_src))[:, 0]

            _, preds = self.io_dec.o(hiddens[-1][:, t]).max(-1)
            preds[eos_yet] = self.fields['trg'].vocab.stoi['<pad>']
            eos_yet = eos_yet | (preds == self.fields['trg'].vocab.stoi['<eos>'])
            outs[:, t + 1] = preds
            if eos_yet.all():
                break

        return outs[:, 1:t+2]

    def beam_search(self, encoding, mask_src=None, width=2, alpha=0.6):  # width: beamsize, alpha: length-norm
        
        encoding = self.decoder.prepare_encoder(encoding)

        W = width
        B, T, C = encoding[0].size()

        # expanding
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(B, W, T, C).contiguous().view(B * W, T, C)
        mask_src = mask_src[:, None, :].expand(B, W, T).contiguous().view(B * W, T)

        T *= self.length_ratio
        outs = encoding[0].new_zeros(B, W, T + 1).long().fill_(self.fields['trg'].vocab.stoi['<pad>'])
        outs[:, :, 0] = self.fields['trg'].vocab.stoi['<init>']

        logps = encoding[0].new_zeros(B, W).float() # scores
        hiddens = [encoding[0].new_zeros(B, W, T, C) for l in range(len(self.decoder.layers) + 1)]

        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].new_zeros(B, W).byte() # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(INF)[:, :, None].expand(B, W, W).contiguous()  # --- BUG, logps < 0 assign INF here 
                                                                                        # --- UPDATE: Aug 9, 2018: BUG again, expand needs contiguous
                                                                                        # --- otherwise everything will become 0.
        eos_mask[:, :, 0] = 0  # batch x beam x beam

        for t in range(T):
            hiddens[0][:, :, t] = self.decoder.prepare_embedding(hiddens[0][:, :, t] + self.io_dec.i(outs[:, :, t], pos=False))

            for l in range(len(self.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.decoder.layers[l].selfattn(x[:, -1:, :], x, x)
                hiddens[l + 1][:, :, t] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l], encoding[l], mask_src)).view(B, W, C)

            # topk2_logps: scores, topk2_inds: top word index at each beam, batch x beam x beam
            topk2_logps = log_softmax(self.io_dec.o(hiddens[-1][:, :, t]))
            topk2_logps[:, :, self.fields['trg'].vocab.stoi['<pad>']] = -INF
            topk2_logps, topk2_inds = topk2_logps.topk(W, dim=-1)

            # mask out the sentences which are finished
            topk2_logps = topk2_logps * (eos_yet[:, :, None].float() * eos_mask + 1 - eos_yet[:, :, None].float())
            topk2_logps = topk2_logps + logps[:, :, None]

            if t == 0:
                logps, topk_inds = topk2_logps[:, 0].topk(W, dim=-1)
            else:
                logps, topk_inds = topk2_logps.view(B, W * W).topk(W, dim=-1)

            topk_beam_inds = topk_inds.div(W)
            topk_token_inds = topk2_inds.view(B, W * W).gather(1, topk_inds)
            eos_yet = eos_yet.gather(1, topk_beam_inds.data)
            
            # logps = logps * (1 - Variable(eos_yet.float()) * 1 / (t + 2)).pow(alpha) # -- bug
            logps = logps * (1 + (eos_yet.float()) * 1 / (t + 1)).pow(alpha)
            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs)).contiguous()
            outs[:, :, t + 1] = topk_token_inds
            topk_beam_inds = topk_beam_inds[:, :, None, None].expand_as(hiddens[0]).contiguous()

            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds)
            eos_yet = eos_yet | (topk_token_inds == self.fields['trg'].vocab.stoi['<eos>'])
            if eos_yet.all():
                return outs[:, 0, 1:]
        return outs[:, 0, 1:]


    def simultaneous_decoding(self, input_stream, mask_stream, agent=None):

        assert self.args.cross_attn_fashion == 'forward', 'currently only forward'
        B, T0 = input_stream.size()
        T = T0 * (1 + self.args.length_ratio)

        # (simulated) input stream
        input_stream = torch.cat([input_stream, input_stream.new_zeros(B, T - T0 + 1)], 1)  # extended 
        mask_stream  = torch.cat([mask_stream,  mask_stream.new_zeros(B, T - T0 + 1)], 1)  # extended
        output_stream = input_stream.new_zeros(B, T + 1).fill_(self.fields['trg'].vocab.stoi['<pad>'])

        # prepare blanks.
        inputs  = input_stream.new_zeros(B, T + 1).fill_(self.fields['src'].vocab.stoi['<init>'])  # inputs
        outputs = input_stream.new_zeros(B, T + 1).fill_(self.fields['trg'].vocab.stoi['<init>'])  # outputs
        
        inputs_mask  = mask_stream.new_zeros(B, T + 1)
        outputs_mask = mask_stream.new_zeros(B, T + 1)

        encoding_outputs = [input_stream.new_zeros(B, T, self.args.d_model).float() 
                            for _ in range(self.args.n_layers + 1)]
        decoding_outputs = [input_stream.new_zeros(B, T, self.args.d_model).float()
                            for _ in range(self.args.n_layers + 1)]

        t_enc = input_stream.new_zeros(B, 1)
        t_dec = input_stream.new_zeros(B, 1)
        eos_yet = input_stream.new_zeros(B, 1).byte()  # stopping mark


        # start real-time translation (please be careful..slow)
        inputs_mask[:, 0]  = 1
        outputs_mask[:, 0] = 1
        
        for t in range(T):

            # encoding
            encoding_outputs[0][:, t:t+1] = self.io_enc.i(inputs[:, t:t+1], pos=False) 
            encoding_outputs[0][:, t:t+1] += positional_encodings_like(encoding_outputs[0][:, t:t+1], t_enc)
            encoding_outputs[0][:, t:t+1] = self.encoder.prepare_embedding(encoding_outputs[0][:, t:t+1])

            for l in range(self.args.n_layers):
                encoding_outputs[l + 1][:, t:t+1] = self.encoder.layers[l].feedforward(
                    self.encoder.layers[l].selfattn(
                        encoding_outputs[l][:, t:t+1], 
                        encoding_outputs[l][:, :t+1], 
                        encoding_outputs[l][:, :t+1], 
                        inputs_mask[:, :t+1]))

            # decoding
            decoding_outputs[0][:, t:t+1] = self.io_dec.i(outputs[:, t:t+1], pos=False)
            decoding_outputs[0][:, t:t+1] += positional_encodings_like(decoding_outputs[0][:, t:t+1], t_dec)
            decoding_outputs[0][:, t:t+1] = self.decoder.prepare_embedding(decoding_outputs[0][:, t:t+1])


            for l in range(self.args.n_layers):
                x = decoding_outputs[l][:, :t+1]
                x = self.decoder.layers[l].selfattn(decoding_outputs[l][:, t:t+1], x, x, outputs_mask[:, :t+1])
                decoding_outputs[l + 1][:, t:t+1] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(
                        x, encoding_outputs[l + 1][:, :t+1], 
                        encoding_outputs[l + 1][:, :t+1], 
                        inputs_mask[:, :t+1]))

            preds = self.io_dec.o(decoding_outputs[-1][:, t:t+1]).max(-1)[1]
            

            # random :: decision
            if agent is None: # random agent
                actions = mask_stream.new_zeros(B, 1).uniform_(0, 1) > 0.9  # 1: write, 0: read
            else:
                actions = agent(encoding_outputs, decoding_outputs, preds)

            # TODO: (optional) if there is no more words left. you cannot read, only write.
            actions = actions | (mask_stream.gather(1, t_enc + 1) == 0)

            # update decoder
            t_dec += actions.long()
            outputs_mask[:, t:t+1] = actions.float()
            outputs_mask[:, t+1] = 1
            preds = preds * actions.long() + outputs[:, t:t+1] * (1 - actions.long())  # if not write, keep the previous word.
            preds[eos_yet] = self.fields['trg'].vocab.stoi['<pad>']
            outputs[:, t+1:t+2] = preds
            
            eos_yet = eos_yet | ((preds == self.fields['trg'].vocab.stoi['<eos>']) & actions)

            # update encoder
            t_enc += 1 - actions.long()
            inputs_mask[:, t+1:t+2] = mask_stream.gather(1, t_enc) * (1 - actions.float())
            inputs[:, t+1:t+2] = input_stream.gather(1, t_enc) 

            # print(actions[0, 0].item(), t_dec[0, 0].item(), 
            #       self.fields['trg'].vocab.itos[outputs[0, t+1].item()])

            # gather data
            output_stream.scatter_(1, t_dec, outputs[:, t+1:t+2])

            if eos_yet.all():
                break

        return output_stream[:, 1:]
        
    def blockwise_parallel_decoding(self, encoding_outputs, mask_stream):
        assert self.args.multi_width > 1, "block-wise parallel decoding only works for multi-step prediction."

        B, T0 = mask_stream.size()
        N  = self.args.multi_width  # multi-step prediction
        T1 = T0 * self.args.length_ratio
        T2 = T1 * N

        # --- encoding --- 
        encoding_outputs = self.decoder.prepare_encoder(encoding_outputs)

        # --- decoding ---

        # prepare blanks
        outputs = mask_stream.new_zeros(B, T2 + 1).long().fill_(self.fields['trg'].vocab.stoi['<init>'])
        outputs_mask = mask_stream.new_zeros(B, T2 + 1)
        decoding_outputs = [mask_stream.new_zeros(B, T2, self.args.d_model).float()
                            for _ in range(self.args.n_layers + 1)]

        t_dec = mask_stream.new_zeros(B, 1).long()           # head
        paces = torch.arange(0, N, device=t_dec.get_device())
        
        eos_yet = mask_stream.new_zeros(B, 1).byte()         # stopping mark

        # start block-wise parallel decoding
        outputs_mask[:, 0] = 1

        for t in range(T1):

            # 0. initialized step.
            if t == 0:
                offset = 1
                pos = t_dec

            else:
                t = 1 + (t - 1) * N
                offset = N
                pos = t_dec + paces[None, :]

            # 1. predict multiple words.
            decoding_outputs[0][:, t: t+offset] = self.io_dec.i(outputs[:, t:t+offset], pos=False)
            decoding_outputs[0][:, t: t+offset] += positional_encodings_like(decoding_outputs[0][:, t:t+offset], pos)
            decoding_outputs[0][:, t: t+offset] = self.decoder.prepare_embedding(decoding_outputs[0][:, t:t+offset])

            for l in range(self.args.n_layers):
                x = decoding_outputs[l][:, :t+offset]
                x = self.decoder.layers[l].selfattn(decoding_outputs[l][:, t:t+offset], x, x, outputs_mask[:, :t+offset])
                decoding_outputs[l + 1][:, t:t+offset] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(
                        x, encoding_outputs[l], encoding_outputs[l], mask_stream))

            curr_outputs = self.io_dec.o(decoding_outputs[-1][:, t:t+offset], full=True).max(-1)[1]
            
            # 2. check with the one-step guess 
            if t == 0:
                prev_outputs = curr_outputs.squeeze(1)
                outputs[:, 1: N+1] = prev_outputs
                t_dec = t_dec + 1   # <bos> is the step we always make first.

            else:
                
                if self.args.dyn == 0 or self.args.exact_match:
                    # block-varying based on exact-matching
                    hits = curr_outputs[:, :-1, 0] == prev_outputs[:, 1:]    # batch_size x (n_step - 1)
                else:
                    # dynamic block-wise based on the model's own prediction.
                    hits = self.io_dec.predict(decoding_outputs[-1][:, t:t+offset-1]).max(-1)[1]

                hits = torch.cat([hits.new_ones(B, 1), hits], 1)         # batch_size x n_step
                new_mask = hits.cumprod(1)                               # batch_size x n_step
                new_index = (new_mask - torch.cat(
                    [new_mask[:, 1:], new_mask.new_zeros(B, 1)], 1)
                    ).max(1)[1]#[:, None, None].expand(B, 1, N)
                new_index_expanded = new_index[:, None, None].expand(B, 1, N)
                new_outputs = curr_outputs.gather(1, new_index_expanded).squeeze(1) # batch_size x n_step
                t_dec = t_dec + new_mask.sum(1, keepdim=True)              # how many steps you actually make

                # 3. check prediction
                new_outputs = new_outputs * (1 - eos_yet.long()) \
                            + self.fields['trg'].vocab.stoi['<eos>'] * eos_yet.long()    # mask dead sentences.
                is_eos = new_outputs[:, 0:1] == self.fields['trg'].vocab.stoi['<eos>'] 
                
                # fatol BUG here: <eos> may come-out earlier as you thought 
                already_eos = prev_outputs.gather(1, new_index[:, None]) == self.fields['trg'].vocab.stoi['<eos>'] 

                eos_yet = eos_yet | is_eos | already_eos  # check if sentence is dead.
                
                prev_outputs = new_outputs

                # 4. make outputs   
                outputs_mask[:, t: t+offset] = new_mask                              # make mask for previous output.
                outputs_mask[:, t+offset: t+offset+1] = 1 - already_eos              # assume the inputs are correct.
                outputs_mask[:, t+offset+1: t+offset*2] = 1 - is_eos | already_eos   # assume the inputs are correct.
                outputs[:, t+offset: t+offset*2] = prev_outputs

                if eos_yet.all():
                    break

        outputs = outputs * outputs_mask.long() + self.fields['trg'].vocab.stoi['<pad>'] * (1 - outputs_mask.long())
        return outputs[:, 1:]

