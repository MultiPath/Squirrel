"""
Fully character-level Transformer with 2D inputs
Experimental use.

Since it changes the basic Attention Mechanism. 
We need to use another file.
"""

import torch
import math

from collections import defaultdict
from namedlist import namedlist
from abc import ABCMeta, abstractmethod
from torch import nn
from torch.nn import functional as F

from model import *

INF = 1e10
TINY = 1e-9

class Message(namedlist('Message', ['data1d', 'mask1d', 'data2d', 'mask2d',
                                    'idx1d', 'idx2d', 'batch_size', 
                                    'max_num_char', 'max_size_word', 'max_num_word',
                                    'causal_mask'])):
    def to2d_(self):
        
        if len(self.data1d.size()) == 3:  # include model dimension
            d_model = self.data1d.size(-1)
            
            self.data2d = self.data1d.new_zeros(self.batch_size * self.max_num_word * self.max_size_word, d_model)
            # self.data2d.resize_(self.batch_size * self.max_num_word * self.max_size_word, d_model).zero_()
            self.data2d[self.idx2d] = self.data1d.view(-1, d_model)[self.idx1d]
            self.data2d = self.data2d.view(self.batch_size, self.max_num_word, self.max_size_word, d_model)
        else:
            self.data2d = self.data1d.new_zeros(self.batch_size * self.max_num_word * self.max_size_word)
            self.data2d[self.idx2d] = self.data1d.view(-1)[self.idx1d]
            self.data2d = self.data2d.view(self.batch_size, self.max_num_word, self.max_size_word)

    def to1d_(self):

        if len(self.data2d.size()) == 4:  # include model dimension
            d_model = self.data2d.size(-1)
            self.data1d = self.data2d.new_zeros(self.batch_size * self.max_num_char, d_model)
            # self.data1d.resize_(self.batch_size * self.max_num_char, d_model).zero_()
            self.data1d[self.idx1d] = self.data2d.view(-1, d_model)[self.idx2d]
            self.data1d = self.data1d.view(self.batch_size, self.max_num_char, d_model)

        else:
            self.data1d = self.data2d.new_zeros(self.batch_size * self.max_num_char)
            self.data1d[self.idx1d] = self.data2d.view(-1)[self.idx2d]
            self.data1d = self.data1d.view(self.batch_size, self.max_num_char)
    
    def refine_mask_(self, mask_index):
        self.mask1d = (self.mask1d.byte() & (self.data1d != mask_index)).float()
        self.mask2d = (self.mask2d.byte() & (self.data2d != mask_index)).float()


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


    def forward(self, message, enc=None, enc_mask=None):

        batch_size, max_num_word, max_size_word, max_num_char = \
        message.batch_size, message.max_num_word, message.max_size_word, message.max_num_char

        # ASSUME our inputs are synced #
    
        # ================= self char-attention ================================== #
        x2d = message.data2d.view(batch_size * max_num_word, max_size_word, -1)
        m2d = message.mask2d.view(batch_size * max_num_word, max_size_word)
        message.data2d = self.selfcharattn(x2d, x2d, x2d, m2d).view(batch_size,max_num_word, max_size_word, -1) 
        message.to1d_()

        # ================= self word-attention ================================= #

        # -- prepare keys & masks --
        x1d_words = message.data2d[:, :, -1]  # batchsize x max_num_word x d_model
        m1d_words = message.mask2d[:, :, -1]  # batchsize x max_num_word

        if self.causal:  # build special masks for causal word attention!!
            m1d_words = message.causal_mask * m1d_words[:, None, :].expand(batch_size, max_num_char, max_num_word)  # () x max_num_word
        message.data1d = self.selfwordattn(message.data1d, x1d_words, x1d_words, m1d_words)

        # ================== cross word-attention =============================== #
        if self.cross:   # only works when doing decoding.
            message.data1d = self.crosswordattn(message.data1d, enc[:, :, -1], enc[:, :, -1], enc_mask[:, :, -1])

        # ================== feed-forward ======================================= #
        message.data1d = self.feedforward(message.data1d)
        message.to2d_()

        return message


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

    def forward(self, message):

        encoding = [message.data2d]
        message.data1d = self.prepare_embedding(message.data1d)
        message.to2d_()

        for layer in self.layers:
            message = layer(message)
            encoding.append(message.data2d)

        message.data2d = encoding
        return message

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

    def forward(self, message, source_message):
        message.data1d = self.prepare_embedding(message.data1d)
        message.to2d_()
            
        source_message.data2d = self.prepare_encoder(source_message.data2d)
        for l, layer in enumerate(self.layers):
            message = layer(message, source_message.data2d[l], source_message.mask2d)

        return message

    def greedy(self, io_dec, source_message):  # direct 1D decoding for characters.

        batch_size, max_num_word, max_size_word, max_num_char = \
        source_message.batch_size, source_message.max_num_word, source_message.max_size_word, source_message.max_num_char
        d_model = self.d_model

        source_message.data2d = self.prepare_encoder(source_message.data2d)
        max_num_char_decoded = max_num_char * self.length_ratio

        char_mask = source_message.data2d[0].new_zeros(batch_size, max_num_char_decoded + 1)
        word_mask = char_mask.new_zeros(batch_size, max_num_char_decoded + 1)
        char_mask[:, 0] = 1
        word_mask[:, 0] = 1

        outputs = char_mask.new_zeros(batch_size, max_num_char_decoded + 1).long().fill_(self.field.vocab.stoi['<init>'])
        word_hiddens = [char_mask.new_zeros(batch_size, max_num_char_decoded, d_model) for _ in range(len(self.layers) + 1)]
        char_hiddens = [char_mask.new_zeros(batch_size, max_num_char_decoded, d_model) for _ in range(len(self.layers))]
        
        word_hiddens[0] = word_hiddens[0] + positional_encodings_like(word_hiddens[0])
        eos_yet = char_mask.new_zeros(batch_size).byte()

        for t in range(max_num_char_decoded):

            # add dropout, etc.
            word_hiddens[0][:, t] = self.prepare_embedding(word_hiddens[0][:, t] + io_dec.i(outputs[:, t], pos=False))

            for l in range(len(self.layers)):
                char_hiddens[l][:, t:t+1] = self.layers[l].selfcharattn(word_hiddens[l][:, t:t+1], word_hiddens[l][:, :t+1], word_hiddens[l][:, :t+1], char_mask[:, :t+1])   
                word_hiddens[l + 1][:, t] = \
                self.layers[l].feedforward(
                    self.layers[l].crosswordattn(
                        self.layers[l].selfwordattn(char_hiddens[l][:, t:t+1], char_hiddens[l][:, :t+1], char_hiddens[l][:, :t+1], word_mask[:, :t+1]), 
                        source_message.data2d[l][:, :, -1], source_message.data2d[l][:, :, -1], source_message.mask2d[:, :, -1]))[:, 0]

            _, preds = io_dec.o(word_hiddens[-1][:, t]).max(-1)

            preds[eos_yet] = self.field.vocab.stoi['<pad>']
            eos_yet = eos_yet | (preds == self.field.vocab.stoi['<eos>'])
            outputs[:, t + 1] = preds

            # fixing the mask
            char_mask = char_mask * (1 - ((outputs[:, t:t+1] == self.field.vocab.stoi['<init>']) | (outputs[:, t:t+1] == self.field.vocab.stoi[' '])).float())  # flush-out char masks
            char_mask[:, t + 1] = 1
            word_mask[:, t + 1] = (outputs[:, t + 1] == self.field.vocab.stoi[' ']).float()

            if eos_yet.all():
                break

        return outputs[:, 1:t+2]


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

    def prepare_masks(self, field, text=None, symbol='<pad>'):
        return (text != self.fields[field].vocab.stoi[symbol]).float()

    def get_messages(self, field, data2d, mask2d, causal=False):

        # get sizes
        batch_size, max_num_word, max_size_word = data2d.size()
        max_num_char = mask2d.view(batch_size, -1).sum(1).long().max().item()

        # source / target pairing
        idx2d = mask2d.view(-1).nonzero().squeeze(1)

        _temp1 = idx2d // (max_num_word * max_size_word)
        _temp2 = (idx2d % (max_num_word * max_size_word) == idx2d[0]).nonzero().squeeze(1)

        idx1d = torch.arange(idx2d.size(0), device=idx2d.get_device()) - _temp2[_temp1] + _temp1 * max_num_char

        # assign data to 1D
        data1d = data2d.new_ones(batch_size * max_num_char)
        mask1d = mask2d.new_zeros(batch_size * max_num_char)
        
        data1d[idx1d] = data2d.view(-1)[idx2d]
        mask1d[idx1d] = mask2d.view(-1)[idx2d]
        
        data1d = data1d.view(batch_size, max_num_char)
        mask1d = mask1d.view(batch_size, max_num_char)

        # prepare a casual mask
        if causal:
            tri_mask = 1 - mask1d.new_ones(max_num_word, max_num_word).triu(1)   # max_num_word x max_num_word
            tri_mask = tri_mask[None, :, None, :].expand(batch_size, max_num_word, max_size_word, max_num_word).contiguous().view(-1, max_num_word)  # (max_num_word  x max_size_word) x max_num_word
            new_mask = tri_mask.new_zeros(batch_size * max_num_char, max_num_word)
            new_mask[idx1d] = tri_mask[idx2d]
            new_mask = new_mask.view(batch_size, max_num_char, max_num_word)    # () x max_num_word
        
        else:
            new_mask = None

        # output message
        message = Message(data1d, mask1d, data2d, mask2d, idx1d, idx2d, batch_size, max_num_char, max_size_word, max_num_word, new_mask)
        message.refine_mask_(self.fields[field].vocab.stoi['<eos>'])  # mask out <EOS> in the input sequence
        return message

    def prepare_data(self, batch):
        source_inputs, target_inputs = batch.src, batch.trg
        source_masks, target_masks = self.prepare_masks('src', source_inputs), self.prepare_masks('trg', target_inputs)

        source_message = self.get_messages('src', source_inputs, source_masks)
        target_message = self.get_messages('trg', target_inputs, target_masks, causal=True)
        return source_message, target_message

    def encoding(self, message):
        message.data1d = self.io_enc.i(message.data1d, pos=True)
        message.to2d_()
        return self.encoder(message)

    def decoding(self, message, source_message, decoding=False, beam=1, alpha=0.6, return_probs=False):

        if (return_probs and decoding) or (not decoding):
            message.data1d = self.io_dec.i(message.data1d, pos=True)
            message.to2d_()
            message = self.decoder(message, source_message)

        if decoding:
            if beam == 1:  # greedy decoding
                output = self.decoder.greedy(self.io_dec, source_message)
            else:
                raise NotImplementedError
            return output
        
        return message


    def forward(self, batch, decoding=False, reverse=True):

        #if info is None:
        info = defaultdict(lambda: 0)

        source_message, target_message = self.prepare_data(batch)
        source_outputs1d = source_message.data1d[:, 1:] 
        target_outputs1d = target_message.data1d[:, 1:]
        target_masks1d   = target_message.mask1d[:, :-1]

        info['sents']  = (source_message.data1d[:, 0] * 0 + 1).sum()
        info['tokens'] = (target_message.mask1d != 0).sum()

        source_message = self.encoding(source_message)
        if not decoding:
            target_message = self.decoding(target_message, source_message)
            loss = self.io_dec.cost(target_outputs1d, target_masks1d, outputs=target_message.data1d[:, :-1], label_smooth=self.args.label_smooth)
            
            for w in loss:
                info['L@' + w] = loss[w]
                if w[0] != '#':
                    info['loss'] = info['loss'] + loss[w]
        else:

            decode_outputs1d = self.decoding(target_message, source_message, decoding=True, return_probs=False)
            
            if reverse:
                source_outputs1d = self.io_enc.reverse(source_outputs1d)
                target_outputs1d = self.io_dec.reverse(target_outputs1d)
                decode_outputs1d = self.io_dec.reverse(decode_outputs1d)

            info['src'] = source_outputs1d
            info['trg'] = target_outputs1d
            info['dec'] = decode_outputs1d
                
        return info