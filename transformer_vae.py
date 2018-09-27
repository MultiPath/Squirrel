from core import *
from transformer import Transformer

class AutoTransformer(Transformer):
    """
    Simple Auto-encoder with the Transformer architecture.
    """

    def __init__(self, src, trg, args):
        super(Transformer, self).__init__()

        self.encoder = Stack(src, args, causal=False, cross=False)
        self.decoder = Stack(trg, args, causal=True,  cross=False) # decoder CANNOT attention to encoder
        
        assert args.multi_width == 1, 'currently only support one-step prediction'

        self.io_dec = IO(trg, args)
        self.io_enc = IO(src, args)
        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.length_ratio = args.length_ratio
        self.fields = {'src': src, 'trg': trg}
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
