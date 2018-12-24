from .core import *
from utils import visualize_attention

class Transformer(Seq2Seq):

    def __init__(self, src, trg, args):
        super().__init__()
        self.encoder = Stack(src, args, causal=args.causal_enc, cross=False, local=args.local_attention)
        self.decoder = Stack(trg, args, causal=True, cross=True)
        
        if args.multi_width > 1:
            self.io_dec = MulIO(trg, args)
            self.io_enc = IO(src, args)
        else:
            self.io_dec = IO(trg, args)
            self.io_enc = IO(src, args)

        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.input_conv = None
        self.length_ratio = args.length_ratio
        self.relative_pos = args.relative_pos
        self.fields = {'src': src, 'trg': trg}
        self.args = args  

        self.attention_maps = None
        self.attention_flag = False
        self.visual_limit = 33

        self.langs = list(set(self.args.src.split(',') + self.args.trg.split(',')))
        for i, lang in enumerate(self.langs):
            self.langs[i] = '<' + lang + '>'

    def trainable_parameters(self):

        def get_params(modules):
            return [p for module in modules for p in module.parameters() if p.requires_grad]
        
        all_param = get_params([self])
        enc_param = get_params([self.encoder, self.io_enc])
        dec_param = get_params([self.decoder, self.io_dec])
        return [all_param, enc_param, dec_param]

    def plot_attention(self, source_inputs, target_inputs, dataflow):
        src = [self.fields[dataflow[0]].init_token] + self.fields[dataflow[0]].reverse(source_inputs)[0].split() + ['<eos>']
        trg = [self.fields[dataflow[1]].init_token] + self.fields[dataflow[1]].reverse(target_inputs)[0].split()
        
        if (len(src) <= self.visual_limit) and (len(trg) <= self.visual_limit):
            
            self.attention_flag = False
            self.attention_maps = []
            for i in range(self.args.n_layers):
                for j in range(self.args.n_heads):
                    fig = visualize_attention(src, src, self.encoder.layers[i].selfattn.layer.attention.p_attn[j].detach()[:len(src), :len(src)])
                    name = 'Self-attention (Enc) L{}/H{}'.format(i, j)
                    self.attention_maps.append((name, fig))

                    fig = visualize_attention(trg, trg, self.decoder.layers[i].selfattn.layer.attention.p_attn[j].detach()[:len(trg), :len(trg)])
                    name = 'Self-attention (Dec) L{}/H{}'.format(i, j)
                    self.attention_maps.append((name, fig))

                    fig = visualize_attention(src, trg, self.decoder.layers[i].crossattn.layer.attention.p_attn[j].detach()[:len(trg), :len(src)])
                    name = 'Cross-attention L{}/H{}'.format(i, j)
                    self.attention_maps.append((name, fig))

    # All in All: forward function for training
    def forward(self, batch, mode='train', reverse=True, dataflow=['src', 'trg'], step=None, lm_only=False):
        
        #if info is None:
        info = defaultdict(lambda: 0)

        source_inputs, source_outputs, source_masks, \
        target_inputs, target_outputs, target_masks = self.prepare_data(batch, dataflow=dataflow)

        info['sents']  = (target_inputs[:, 0] * 0 + 1).sum()
        info['tokens'] = (target_masks != 0).sum()
        info['max_att'] = info['sents'] * max((source_inputs[0, :] * 0 + 1).sum() ** 2,
                                              (target_inputs[0, :] * 0 + 1).sum() ** 2)
        
        # encoding: if first dataflow is empty, then ignore the encoder.
        if not lm_only:
            encoding_inputs  = self.io_enc.i(source_inputs, pos=(not self.relative_pos))
            encoding_outputs = self.encoder(encoding_inputs, source_masks)
        else:
            encoding_outputs, source_masks = None, None

        if self.training:
            label_smooth = self.args.label_smooth
        else:
            label_smooth = 0.0

        if mode == 'train':
            # Maximum Likelihood Training (with label smoothing trick)
            decoding_inputs = self.io_dec.i(target_inputs, pos=(not self.relative_pos))
            decoding_outputs = self.decoder(decoding_inputs, target_masks, encoding_outputs, source_masks)
            loss = self.io_dec.cost(target_outputs, target_masks, outputs=decoding_outputs[-1], label_smooth=label_smooth)
            
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

            # Source side Language Model (optional, only works for causal-encoder)
            if self.args.encoder_lm and self.args.causal_enc:
                loss_lm = self.io_enc.cost(source_outputs, source_masks, outputs=encoding_outputs[-1])
                for w in loss_lm:
                    info['L@' + w] = loss_lm[w]
                    if w[0] != '#':
                        info['loss'] = info['loss'] + loss_lm[w]

            if self.attention_flag:
                self.plot_attention(source_inputs, target_inputs, dataflow)
                            
        else:
            # Decoding (for evaluation)
            if self.args.multi_width > 1: # -- the newly introduced block-wise decoding --
                assert self.args.beam_size == 1, 'block-wise decoding only works for greedy decoding (for now).' 

                translation_outputs = self.blockwise_parallel_decoding(encoding_outputs, source_masks, field=dataflow[1])
            else:
                if self.args.beam_size == 1:
                    translation_outputs = self.greedy_decoding(encoding_outputs, source_masks, field=dataflow[1])
                else:
                    translation_outputs = self.beam_search(encoding_outputs, source_masks, self.args.beam_size, self.args.alpha, field=dataflow[1])

            if reverse:
                source_outputs = self.fields[dataflow[0]].reverse(source_outputs)
                target_outputs = self.fields[dataflow[1]].reverse(target_outputs)
                
                # specially for multi_step decoding #
                if self.args.multi_width > 1:
                    translation_outputs, saved_time, pred_acc, decisions = self.fields[dataflow[1]].reverse(translation_outputs, width=self.args.multi_width, return_saved_time=True)
                    
                    info['saved_time'] = saved_time
                    info['pred_acc'] = pred_acc
                    info['decisions'] = decisions
                
                else:
                    translation_outputs = self.fields[dataflow[1]].reverse(translation_outputs)

            info['src'] = source_outputs
            info['trg'] = target_outputs
            info['dec'] = translation_outputs
        
        return info

    def greedy_decoding(self, encoding=None, mask_src=None, T=None, field='trg'):

        encoding = self.decoder.prepare_encoder(encoding)
        if T is None:
            T = encoding[0].size()[1]
        B, C = encoding[0].size()[0], encoding[0].size()[-1]  # batch_size, decoding-length, size
        T *= self.length_ratio

        outs = encoding[0].new_zeros(B, T + 1).long().fill_(self.fields[field].vocab.stoi[self.fields[field].init_token])
        hiddens = [encoding[0].new_zeros(B, T, C) for l in range(len(self.decoder.layers) + 1)]

        if not self.relative_pos:
            hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])  # absolute positions
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
            preds[eos_yet] = self.fields[field].vocab.stoi['<pad>']
            eos_yet = eos_yet | (preds == self.fields[field].vocab.stoi['<eos>'])
            outs[:, t + 1] = preds
            if eos_yet.all():
                break

        return outs[:, 1:t+2]

    def beam_search(self, encoding, mask_src=None, width=2, alpha=0.6, T=None, field='trg'):  # width: beamsize, alpha: length-norm
        
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

        logps = encoding[0].new_zeros(B, W).float() # scores
        hiddens = [encoding[0].new_zeros(B, W, T, C) for l in range(len(self.decoder.layers) + 1)]

        if not self.relative_pos:
            hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])  # absolute positions

        eos_yet = encoding[0].new_zeros(B, W).byte() # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(INF)[:, :, None].expand(B, W, W).contiguous()  # --- BUG, logps < 0 assign INF here 
                                                                                        # --- UPDATE: Aug 9, 2018: BUG again, expand needs contiguous
                                                                                        # --- otherwise everything will become 0.
        eos_mask[:, :, 0] = 0  # batch x beam x beam

        for t in range(T):
            hiddens[0][:, :, t] = self.decoder.prepare_embedding(hiddens[0][:, :, t] + self.io_dec.i(outs[:, :, t], pos=False))

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.decoder.layers[l].selfattn(x[:, -1:, :], x, x)
                hiddens[l + 1][:, :, t] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l], encoding[l], mask_src)).view(B, W, C)

            # topk2_logps: scores, topk2_inds: top word index at each beam, batch x beam x beam
            topk2_logps = log_softmax(self.io_dec.o(hiddens[-1][:, :, t]))
            topk2_logps[:, :, self.fields[field].vocab.stoi['<pad>']] = -INF
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
            eos_yet = eos_yet | (topk_token_inds == self.fields[field].vocab.stoi['<eos>'])
            if eos_yet.all():
                return outs[:, 0, 1:]
        return outs[:, 0, 1:]

    def simultaneous_decoding(self, input_stream, mask_stream, agent=None):

        assert self.args.cross_attn_fashion == 'forward', 'currently only forward'
        assert (not self.relative_pos), 'currently only support absoulte positions'

        B, T0 = input_stream.size()
        T = T0 * (1 + self.args.length_ratio)

        # (simulated) input stream
        input_stream = torch.cat([input_stream, input_stream.new_zeros(B, T - T0 + 1)], 1)  # extended 
        mask_stream  = torch.cat([mask_stream,  mask_stream.new_zeros(B, T - T0 + 1)], 1)  # extended
        output_stream = input_stream.new_zeros(B, T + 1).fill_(self.fields['trg'].vocab.stoi['<pad>'])

        # prepare blanks.
        inputs  = input_stream.new_zeros(B, T + 1).fill_(self.fields['src'].vocab.stoi[self.fields['src'].init_token])  # inputs
        outputs = input_stream.new_zeros(B, T + 1).fill_(self.fields['trg'].vocab.stoi[self.fields['trg'].init_token])  # outputs
        
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

            # gather data
            output_stream.scatter_(1, t_dec, outputs[:, t+1:t+2])

            if eos_yet.all():
                break

        return output_stream[:, 1:]
        
    def blockwise_parallel_decoding(self, encoding_outputs, mask_stream, field='trg'):
        
        assert self.args.multi_width > 1, "block-wise parallel decoding only works for multi-step prediction."
        assert (not self.relative_pos), 'currently only support absoulte positions'

        B, T0 = mask_stream.size()
        N  = self.args.multi_width  # multi-step prediction
        T1 = T0 * self.args.length_ratio
        T2 = T1 * N

        # --- encoding --- 
        encoding_outputs = self.decoder.prepare_encoder(encoding_outputs)

        # --- decoding ---

        # prepare blanks
        outputs = mask_stream.new_zeros(B, T2 + 1).long().fill_(self.fields[field].vocab.stoi[self.fields[field].init_token])
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
                            + self.fields[field].vocab.stoi['<eos>'] * eos_yet.long()    # mask dead sentences.
                is_eos = new_outputs[:, 0:1] == self.fields[field].vocab.stoi['<eos>'] 
                
                # fatol BUG here: <eos> may come-out earlier as you thought 
                already_eos = prev_outputs.gather(1, new_index[:, None]) == self.fields[field].vocab.stoi['<eos>'] 

                eos_yet = eos_yet | is_eos | already_eos  # check if sentence is dead.
                
                prev_outputs = new_outputs

                # 4. make outputs   
                outputs_mask[:, t: t+offset] = new_mask                              # make mask for previous output.
                outputs_mask[:, t+offset: t+offset+1] = 1 - already_eos              # assume the inputs are correct.
                outputs_mask[:, t+offset+1: t+offset*2] = 1 - is_eos | already_eos   # assume the inputs are correct.
                outputs[:, t+offset: t+offset*2] = prev_outputs

                if eos_yet.all():
                    break

        outputs = outputs * outputs_mask.long() + self.fields[field].vocab.stoi['<pad>'] * (1 - outputs_mask.long())
        return outputs[:, 1:]

