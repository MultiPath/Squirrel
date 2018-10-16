from .core import *
from .transformer import Transformer

class AutoTransformer2(Transformer):
    """
    Simple Auto-encoder with the Transformer architecture.
    """

    def __init__(self, src, trg, args):
        super(Transformer, self).__init__()
        
        self.encoder = Stack(src, args, causal=False, cross=False)
        self.decoder = Stack(trg, args, causal=True,  cross=True)  # decoder CANNOT attention to encoder
        
        self.proj_i = Linear(args.d_model, args.d_model)
        self.proj_o = Linear(args.d_model, args.d_model * args.n_proj_layers)

        assert args.multi_width == 1, 'currently only support one-step prediction'

        self.io_dec = IO(trg, args)
        self.io_enc = IO(src, args)
        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.length_ratio = args.length_ratio
        self.fields = {'src': src, 'trg': trg}
        self.args = args  

        self.pool = args.pool
        self.noise = args.latent_noise
        self.counter = 0
        self.count_base = 200000
    
        # space projector (cannot be trained jointly)
        self.src2trg = Linear(args.d_model, args.d_model)
        self.trg2src = Linear(args.d_model, args.d_model)
        # self.src2trg = FeedForward(args.d_model, args.d_hidden) # Linear(args.d_model, args.d_model)
        # self.trg2src = FeedForward(args.d_model, args.d_hidden) # Linear(args.d_model, args.d_model)

    def trainable_parameters(self):
        param_ae = [self.encoder, self.decoder, self.proj_i, self.proj_o, self.io_enc, self.io_dec]
        param_mt = [self.src2trg, self.trg2src]
        param_ae = [p for module in param_ae for p in module.parameters()]
        param_mt = [p for module in param_mt for p in module.parameters()]
        return [param_ae, param_mt, self.parameters()]

    # All in All: forward function for training
    def forward(self, batch, decoding=False, reverse=True, dataflow=['src', 'src'], noise_level=None):
        
        #if info is None:
        info = defaultdict(lambda: 0)
        source_inputs, source_outputs, source_masks, \
        target_inputs, target_outputs, target_masks = self.prepare_data(batch, dataflow=dataflow, noise=noise_level)
        batch_size = source_inputs.size(0)

        info['sents']  = (target_inputs[:, 0] * 0 + 1).sum()
        info['tokens'] = (target_masks != 0).sum()

        # in some extreme case.
        if info['sents'] == 0:
            return info

        # encoding
        if self.pool == 'mean':
            encoding_outputs = (self.encoder(self.io_enc.i(source_inputs, pos=True), source_masks)[-1] \
                                * source_masks[:, :, None]).sum(1) / source_masks.sum(1, keepdim=True)  # batch_size x d_model'
        elif self.pool == 'max':
            encoding_outputs = (self.encoder(self.io_enc.i(source_inputs, pos=True), source_masks)[-1] - (1 - source_masks[:, :, None]) * INF).max(1)[0]
        else:
            raise NotImplementedError

        # hidden representations
        hidden = torch.tanh(self.proj_i(encoding_outputs))
        
        # space "Linear" projection.
        if (dataflow[0] == 'src') and (dataflow[1] == 'trg'):
            hidden = self.src2trg(hidden)
        elif (dataflow[0] == 'trg') and (dataflow[1] == 'src'):
            hidden = self.trg2src(hidden)

        # (optional) add latent noise
        if self.training and (self.noise > 0):
            hidden = hidden + hidden.new_zeros(*hidden.size()).normal_(0, self.noise)

        # decoding 
        decoding_inputs = self.proj_o(hidden).view(batch_size, self.args.n_proj_layers, self.args.d_model)  # batch x d_model
        decoding_inputs = [decoding_inputs for t in range(self.args.n_layers)]

        if not decoding:

            self.counter += 1

            # Maximum Likelihood Training (with label smoothing trick)
            decoding_inputs_embed = self.io_dec.i(target_inputs)
            decoding_outputs = self.decoder(decoding_inputs_embed, target_masks, decoding_inputs)  # no source side attention. all information compressed in "decoding_inputs_embed".
            loss = self.io_dec.cost(target_outputs, target_masks, outputs=decoding_outputs[-1], label_smooth=self.args.label_smooth)
            
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

        else:
            # Decoding (for evaluation)

            if self.args.beam_size == 1:
                translation_outputs = self.greedy_decoding(decoding_inputs, T = source_masks.size(1), field=dataflow[1])
            else:
                raise NotImplementedError
                #translation_outputs = self.beam_search(encoding_outputs, source_masks, self.args.beam_size, self.args.alpha)

            if reverse:
                source_outputs = self.fields[dataflow[0]].reverse(source_outputs)
                target_outputs = self.fields[dataflow[1]].reverse(target_outputs)
                translation_outputs = self.fields[dataflow[1]].reverse(translation_outputs)

            info['src'] = source_outputs
            info['trg'] = target_outputs
            info['dec'] = translation_outputs
        
        return info

        
# ------- Old Codes --------- #
class AutoTransformer(Transformer):
    """
    Simple Auto-encoder with the Transformer architecture.
    """

    def __init__(self, src, trg, args):
        super(Transformer, self).__init__()

        self.encoder = Stack(src, args, causal=False, cross=False)
        self.decoder = Stack(trg, args, causal=True,  cross=False) # decoder CANNOT attention to encoder
        
        if args.variational:
            raise NotImplementedError

        self.proj_i = Linear(args.d_model, args.d_model)
        self.proj_o = Linear(args.d_model, args.d_model)

        assert args.multi_width == 1, 'currently only support one-step prediction'

        self.io_dec = IO(trg, args)
        self.io_enc = IO(src, args)
        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.length_ratio = args.length_ratio
        self.fields = {'src': src, 'trg': trg}
        self.pool = args.pool
        self.args = args  

    
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
        if self.pool == 'mean':
            encoding_outputs = (self.encoder(self.io_enc.i(source_inputs, pos=True), source_masks)[-1] \
                                * source_masks[:, :, None]).sum(1) / source_masks.sum(1, keepdim=True)  # batch_size x d_model'
        elif self.pool == 'max':
            encoding_outputs = (self.encoder(self.io_enc.i(source_inputs, pos=True), source_masks)[-1] - (1 - source_masks[:, :, None]) * INF).max(1)
        else:
            raise NotImplementedError

        # hidden representations
        hidden = torch.tanh(self.proj_i(encoding_outputs))

        # decoding 
        decoding_inputs = self.proj_o(hidden)  # batch x d_model

        if not decoding:
            # Maximum Likelihood Training (with label smoothing trick)
            decoding_inputs_embed = self.io_dec.i(target_inputs)

            if self.args.ae_func == 'first_step':
                decoding_inputs_embed[:, 0, :] = decoding_inputs_embed[:, 0, :] + decoding_inputs
            elif self.args.ae_func == 'all_steps':
                decoding_inputs_embed = decoding_inputs_embed + decoding_inputs[:, None, :]
            else:
                raise NotImplementedError
            
            decoding_outputs = self.decoder(decoding_inputs_embed, target_masks)  # no source side attention. all information compressed in "decoding_inputs_embed".
            loss = self.io_dec.cost(target_outputs, target_masks, outputs=decoding_outputs[-1], label_smooth=self.args.label_smooth)
            
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

        else:
            # Decoding (for evaluation)

            if self.args.beam_size == 1:
                translation_outputs = self.greedy_decoding(decoding_inputs, source_masks.size(1))
            else:
                raise NotImplementedError
                #translation_outputs = self.beam_search(encoding_outputs, source_masks, self.args.beam_size, self.args.alpha)

            if reverse:
                source_outputs = self.io_enc.reverse(source_outputs)
                target_outputs = self.io_dec.reverse(target_outputs)
                translation_outputs = self.io_dec.reverse(translation_outputs)

            info['src'] = source_outputs
            info['trg'] = target_outputs
            info['dec'] = translation_outputs
        
        return info

    def greedy_decoding(self, start, T=100):

        B, C = start.size()  # batch_size, decoding-length, size
        T = T * self.length_ratio

        outs = start.new_zeros(B, T + 1).long().fill_(self.fields['trg'].vocab.stoi['<init>'])
        hiddens = [start.new_zeros(B, T, C) for l in range(len(self.decoder.layers) + 1)]

        if self.args.ae_func == 'first_step':
            hiddens[0][:, 0, :] = hiddens[0][:, 0, :] + start
        elif self.args.ae_func == 'all_steps':
            hiddens[0] = hiddens[0] + start[:, None, :]
        else:
            raise NotImplementedError

        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = start.new_zeros(B).byte()

        for t in range(T):

            # add dropout, etc.
            hiddens[0][:, t] = self.decoder.prepare_embedding(hiddens[0][:, t] + self.io_dec.i(outs[:, t], pos=False))
            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :t+1]
                x = self.decoder.layers[l].selfattn(hiddens[l][:, t:t+1], x, x)   # we need to make the dimension 3D
                hiddens[l + 1][:, t] = self.decoder.layers[l].feedforward(x)[:, 0]

            _, preds = self.io_dec.o(hiddens[-1][:, t]).max(-1)
            preds[eos_yet] = self.fields['trg'].vocab.stoi['<pad>']
            eos_yet = eos_yet | (preds == self.fields['trg'].vocab.stoi['<eos>'])
            outs[:, t + 1] = preds
            if eos_yet.all():
                break

        return outs[:, 1:t+2]

