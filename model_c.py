"""
Fully character-level Transformer with 2D inputs
Experimental use.

Since it changes the basic Attention Mechanism. 
We need to use another file.
"""

import torch
import math

from collections import defaultdict
from abc import ABCMeta, abstractmethod
from torch import nn
from torch.nn import functional as F

from model import *

INF = 1e10
TINY = 1e-9


class CharWordBlock(nn.Module):
    
    def __init__(self, args, causal=False, cross=False):
        super().__init__()
        self.selfcharattn = ResidualBlock(
            MultiHead2(
                args.d_model, args.d_model, args.n_heads,
                args.drop_ratio, causal),
            args.d_model, args.drop_ratio, order='tdan')

        self.selfwordattn = ResidualBlock(
            MultiHead2(
                args.d_model, args.d_model, args.n_heads,
                args.drop_ratio, False),
            args.d_model, args.drop_ratio, order='tdan')

        if cross:
            self.crosswordattn = ResidualBlock(
            MultiHead2(
                args.d_model, args.d_model, args.n_heads,
                args.drop_ratio),  # only noisy when doing cross-attention
            args.d_model, args.drop_ratio, order='tdan')

        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden, args.drop_ratio),
            args.d_model, args.drop_ratio, order='tdan')

        self.cross = cross
        self.causal = causal

    def forward(self, x2d, m2d, idx1d, idx2d, max_char, encoding=None, mask_src=None, reshape_back=True):

        # ================= self char-attention ================================== #
        batch_size, max_num_word, max_size_word, d_model = x2d.size()
        x2d = x2d.view(-1, max_size_word, d_model)
        x2d = self.selfcharattn(x2d, x2d, x2d, m2d.view(-1, max_size_word)).view(batch_size, max_num_word, max_size_word, d_model) 

        # ================= self word-attention ================================= #

        # -- prepare query --
        x1d = x2d.new_zeros(batch_size * max_char, d_model)
        x1d[idx1d] = x2d.view(-1, d_model)[idx2d]
        x1d = x1d.view(batch_size, -1, d_model).contiguous()  # batchsize x max_char x d_model

        # -- prepare keys & masks --
        x1d_words = x2d[:, :, -1]  # batchsize x max_num_word x d_model
        m1d_words = m2d[:, :, -1]  # batchsize x max_num_word

        if self.causal:  # build special masks for causal word attention!!
            
            tri_mask = 1 - m1d_words.new_ones(max_num_word, max_num_word).triu(1)   # max_num_word x max_num_word
            tri_mask = tri_mask[None, :, None, :].expand(batch_size, max_num_word, max_size_word, max_num_word).contiguous().view(-1, max_num_word)  # (max_num_word  x max_size_word) x max_num_word
            new_mask = tri_mask.new_zeros(batch_size * max_char, max_num_word)
            new_mask[idx1d] = tri_mask[idx2d]
            m1d_words = new_mask.view(batch_size, max_char, max_num_word) * m1d_words[:, None, :].expand(batch_size, max_char, max_num_word)  # () x max_num_word

        # print(x1d.size(), x1d_words.size(), m1d_words.size())
        x1d = self.selfwordattn(x1d, x1d_words, x1d_words, m1d_words)

        # ================== cross word-attention =============================== #
        
        if self.cross:   # only works when doing decoding.
            encoding = encoding[:, :, -1]
            mask_src = mask_src[:, :, -1]
            x1d = self.crosswordattn(x1d, encoding, encoding, mask_src)

        # ================== feed-forward ======================================= #
        x1d = self.feedforward(x1d)

        # ================== return to 2D ======================================= #
        if reshape_back:
            x2d = x1d.new_zeros(batch_size * max_num_word * max_size_word, d_model)
            x2d[idx2d] = x1d.view(-1, d_model)[idx1d]
            x2d = x2d.view(batch_size, max_num_word, max_size_word, d_model).contiguous()

            return x2d

        else:
            
            return x1d  


class EncoderC(Encoder):

    def __init__(self, field, args):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [CharWordBlock(args) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)
        
        if args.normalize_emb:
            self.layernorm = LayerNorm(args.d_model)

        self.field = field
        self.d_model = args.d_model
        self.share_embeddings = args.share_embeddings
        self.normalize_emb = args.normalize_emb

    def forward(self, x, mask, idx1d, idx2d, max_char):

        encoding = [x]
        x = self.prepare_embedding(x)
        
        for layer in self.layers:
            x = layer(x, mask, idx1d, idx2d, max_char)
            encoding.append(x)

        return encoding

class DecoderC(Decoder):

    def __init__(self, field, args, causal=True):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [CharWordBlock(args, causal, cross=True)
            for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)

        if args.normalize_emb:
            self.layernorm = LayerNorm(args.d_model)

        self.d_model = args.d_model
        self.field = field
        self.length_ratio = args.length_ratio
        self.cross_attn_fashion = args.cross_attn_fashion
        self.normalize_emb = args.normalize_emb

    def forward(self, x, mask_trg, idx1d, idx2d, max_char, encoding=None, mask_src=None):
        x = self.dropout(x)
        if self.normalize_emb:
            x = self.layernorm(x)
            
        encoding = self.prepare_encoder(encoding)
        L = len(self.layers)
        for l, (layer, enc) in enumerate(zip(self.layers, encoding)):
            if l < (L - 1):
                x = layer(x, mask_trg, idx1d, idx2d, max_char, enc, mask_src)
            else:
                # Important: we do not need to reshape back from last decoder layer
                x = layer(x, mask_trg, idx1d, idx2d, max_char, enc, mask_src, reshape_back=False)

        return x



class TransformerC(Transformer):

    def __init__(self, src, trg, args):
        super(Transformer, self).__init__()

        self.encoder = EncoderC(src, args)
        self.decoder = DecoderC(trg, args, causal=True)

        self.io_dec = IO(trg, args)
        self.io_enc = IO(src, args)

        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.fields = {'src': src, 'trg': trg}
        self.args = args

        # decode or not:
        self.decode = False

    def prepare_masks(self, inputs, symbol='<pad>', masks=None):
        """
        CAUTION: this function is working differently as in Transformer
        """
        field, text = inputs
        
        if masks is None:
            masks = ((text.data != self.fields[field].vocab.stoi[symbol])).float()
        else:
            masks = (masks.byte() & (text.data != self.fields[field].vocab.stoi[symbol])).float()
        return masks

    def trans_2d_to_1d(self, data2d, mask2d):

        # get sizes
        batch_size, max_num_word, max_size_word = data2d.size()
        max_num_char = mask2d.view(batch_size, -1).sum(1).long().max().item()

        # everything put in 1D
        data1d = data2d.new_ones(batch_size * max_num_char)
        mask1d = mask2d.new_zeros(batch_size * max_num_char)

        # source / target pairing
        idx2d = mask2d.view(-1).nonzero().squeeze(1)

        _temp1 = idx2d // (max_num_word * max_size_word)
        _temp2 = (idx2d % (max_num_word * max_size_word) == idx2d[0]).nonzero().squeeze(1)

        idx1d = torch.arange(idx2d.size(0), device=idx2d.get_device()) - _temp2[_temp1] + _temp1 * max_num_char

        # assign data to 1D
        data1d[idx1d] = data2d.view(-1)[idx2d]
        mask1d[idx1d] = mask2d.view(-1)[idx2d]
        return data1d.view(batch_size, max_num_char).contiguous(), mask1d.view(batch_size, max_num_char), idx2d, idx1d, max_num_char      

    def prepare_data(self, batch):
        source_inputs, target_inputs = batch.src, batch.trg
        source_masks, target_masks = self.prepare_masks(('src', source_inputs)), self.prepare_masks(('trg', target_inputs))

        source_inputs1d, source_mask1d, source_idx2d, source_idx1d, source_max_char = self.trans_2d_to_1d(source_inputs, source_masks)
        target_inputs1d, target_mask1d, target_idx2d, target_idx1d, target_max_char = self.trans_2d_to_1d(target_inputs, target_masks)

        # refine the masks (because we need to maskout <eos> in the input side.)
        source_masks = self.prepare_masks(('src', source_inputs), '<eos>', source_masks)
        target_masks = self.prepare_masks(('src', target_inputs), '<eos>', target_masks)

        source_outputs1d, source_mask1d = source_inputs1d[:, 1:], source_mask1d[:, 1:]
        target_outputs1d, target_mask1d = target_inputs1d[:, 1:], target_mask1d[:, 1:]

        return source_inputs, source_outputs1d, source_masks, source_mask1d, \
            target_inputs, target_outputs1d, target_masks, target_mask1d, \
            source_idx1d, source_idx2d, source_max_char, target_idx1d, target_idx2d, target_max_char

    
    def encoding(self, encoder_inputs, encoder_masks, idx1d, idx2d, max_char):
        return self.encoder(self.io_enc.i(encoder_inputs, pos=True), encoder_masks, idx1d, idx2d, max_char)

    def decoding(self, decoder_inputs, decoder_masks, idx1d, idx2d, max_char,
                encoding_outputs, encoder_masks,
                decoding=False, beam=1, alpha=0.6, return_probs=False):

        if (return_probs and decoding) or (not decoding):
            out = self.decoder(self.io_dec.i(decoder_inputs, pos=True), decoder_masks, idx1d, idx2d, max_char, encoding_outputs, encoder_masks)

        if decoding:
            raise NotImplementedError

        return out


    def forward(self, batch, decoding=False, reverse=True):

        #if info is None:
        info = defaultdict(lambda: 0)

        source_inputs, source_outputs1d, source_masks, source_masks1d, \
        target_inputs, target_outputs1d, target_masks, target_masks1d, \
        source_idx1d, source_idx2d, source_max_char, \
        target_idx1d, target_idx2d, target_max_char = self.prepare_data(batch)

        info['sents']  = (target_outputs1d[:, 0] * 0 + 1).sum()
        info['tokens'] = (target_masks1d != 0).sum()

        # # in some extreme case.
        # if info['sents'] == 0:
        #     return info

        # encoding
        encoding_outputs = self.encoding(source_inputs, source_masks, source_idx1d, source_idx2d, source_max_char)

        if not decoding:

            # Maximum Likelihood Training (with label smoothing trick)
            decoding_outputs = self.decoding(target_inputs, target_masks, target_idx1d, target_idx2d, target_max_char, encoding_outputs, source_masks)[:, :-1]   
            loss = self.io_dec.cost(target_outputs1d, target_masks1d, outputs=decoding_outputs, label_smooth=self.args.label_smooth)
            
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]

            

        else:

            raise NotImplementedError
    
        
        return info