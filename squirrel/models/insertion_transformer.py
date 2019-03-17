"""
Re-implementation (with MORE extensions) of Google's InsertionTransformer

Stern M, Chan W, Kiros J, Uszkoreit J. 
Insertion Transformer: Flexible Sequence Generation via Insertion Operations. 
arXiv preprint arXiv:1902.03249. 2019 Feb 8.
"""

import time
from collections import defaultdict

import torch

from squirrel.models import register_new_model
from squirrel.models.core import (IO, Attention, Linear, Seq2Seq, Stack,
                                  log_softmax)
from squirrel.utils import INF, NegativeDistanceScore

try:
    from squirrel.metrics.fast_editdistance import suggested_ed2_path
except ImportError:
    from squirrel.metrics.editdistance import suggested_ed2_path


@register_new_model('InsertionTransformer')
class InsertionTransformer(Seq2Seq):
    """
    InsertionTransformer with slot-termination
    """

    def __init__(self, src, trg, args):
        super().__init__()

        self.length_ratio = args.length_ratio
        self.relative_pos = args.relative_pos
        self.source_relative_pos = args.relative_pos
        self.fields = {'src': src, 'trg': trg}
        self.args = args
        self.score_fn = NegativeDistanceScore()
        self.terminal = self.fields['trg'].pad_token
        self.index_t = self.fields['trg'].vocab.stoi[self.terminal]
        self.encoder = Stack(
            src,
            args,
            causal=False,
            cross=False,
            relative_pos=self.source_relative_pos)

        self.decoder = Stack(
            trg,
            args,
            causal=False,
            cross=True,
            relative_pos=self.relative_pos)

        self.io_enc = IO(src, args)
        self.io_dec = IO(trg, args)
        self.trans_c = Linear(args.d_model * 2, args.d_embedding)
        self.trans_l = Linear(args.d_model * 2, args.d_model)
        self.position_query = Linear(args.d_model, 1, bias=False)
        self.pointer = Attention(
            args.d_model, 1, 0.0, False, relative_pos=False)

        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.langs = list(
            set(self.args.src.split(',') + self.args.trg.split(',')))
        for i, lang in enumerate(self.langs):
            self.langs[i] = '<' + lang + '>'

    def generate_training_labels(self,
                                 batch_noisy,
                                 batch,
                                 input_tensor,
                                 tau=None,
                                 get_delete_labels=True):

        V = len(self.fields['trg'].vocab)
        B = input_tensor.size(0)
        T = input_tensor.size(1)

        batch = [[self.fields['trg'].vocab.stoi[w] for w in b] for b in batch]
        batch_noisy = [[self.fields['trg'].vocab.stoi[w] for w in b]
                       for b in batch_noisy]

        full_labels = suggested_ed2_path(batch_noisy, batch, self.index_t)
        insert_labels = [a[:-1] for a in full_labels]
        delete_labels = [b[-1] for b in full_labels]

        # numericalize2
        # insert_labels = [[(w, j, self.score_fn(k, len(label), tau))
        #                   for j, label in enumerate(labels)
        #                   for k, w in enumerate(label)]
        #                  for labels in insert_labels]
        # insert_maxlen = max([len(labels) for labels in insert_labels])

        # insert_word, insert_location, insert_weights = [
        #     torch.tensor([[l[k] for l in labels] +
        #                   [0.0 for _ in range(insert_maxlen - len(labels))]
        #                   for labels in insert_labels],
        #                  device=input_tensor.get_device()) for k in range(3)
        # ]
        # insert_word = insert_word.long()
        # insert_location = insert_location.long()

        # return insert_word, insert_location, insert_weights

        # numericalize1
        insert_label_tensors = input_tensor.new_zeros(B * (T - 1) * V).float()
        insert_index, insert_labels = zip(
            *[(w + (j + i * (T - 1)) * V, self.score_fn(k, len(label), tau))
              for i, labels in enumerate(insert_labels)
              for j, label in enumerate(labels) for k, w in enumerate(label)])

        insert_index, insert_labels = [
            torch.tensor(list(a), device=input_tensor.get_device())
            for a in [insert_index, insert_labels]
        ]

        insert_label_tensors.scatter_(0, insert_index.long(), insert_labels)
        insert_label_tensors = insert_label_tensors.view(B, T - 1, V)
        terminal_tensors = (
            insert_label_tensors[:, :, self.index_t] == 0).float()

        if get_delete_labels:
            # add pads
            delete_labels = [
                labels + [0 for _ in range(T - 2 - len(labels))]
                for labels in delete_labels
            ]
            delete_label_tensors = torch.tensor(
                delete_labels, device=input_tensor.get_device())
            return insert_label_tensors, terminal_tensors, delete_label_tensors
        return insert_label_tensors, terminal_tensors

    def prepare_field(self, batch, field):
        data = getattr(batch, field)
        original = getattr(batch, field + '_original')

        if hasattr(batch, field + '_n'):
            inputs = getattr(batch, field + '_n')
            outputs = data
            masks = self.prepare_masks((field, inputs))

        else:
            inputs = data
            outputs = data
            masks = self.prepare_masks((field, inputs))

        return inputs, outputs, masks, original

    def prepare_content_location_representations(self,
                                                 decoding_outputs,
                                                 target_masks=None):

        slots_outputs = torch.cat(
            [decoding_outputs[:, :-1], decoding_outputs[:, 1:]], 2)
        slots_outputs_contents = self.trans_c(slots_outputs)
        slots_outputs_location = self.trans_l(slots_outputs)

        if target_masks is None:
            return slots_outputs_contents, slots_outputs_location
        slots_masks = target_masks[:, 1:]
        slots_targets = slots_masks / slots_masks.sum(-1, keepdim=True)
        return slots_outputs_contents, slots_outputs_location, slots_masks, slots_targets

    def prepare_position_query_vectors(self, batch_size=1):
        return self.position_query.weight.expand(
            batch_size, 1, self.args.d_model).contiguous()

    def forward(self, batch, mode='train', dataflow=['src', 'trg'], step=None):

        if not (dataflow[0] == 'src' and dataflow[1] == 'trg'):
            raise NotImplementedError(
                'currently in default flow from source to target.')

        info = defaultdict(lambda: 0)
        source_inputs, source_outputs, source_masks, source_original, \
            target_inputs, target_outputs, target_masks, target_original = self.prepare_data(batch, dataflow=dataflow)
        batch_size = source_inputs.size(0)

        #  vocab_size = len(self.fields['trg'].vocab)

        info['sents'] = (target_outputs[:, 0] * 0 + 1).sum()
        info['tokens'] = (target_masks != 0).sum()
        info['full_tokens'] = (target_outputs != self.index_t).sum()

        if dataflow[0] is not None:
            encoding_inputs = self.io_enc.i(
                source_inputs, pos=(not self.relative_pos))
            encoding_outputs = self.encoder(encoding_inputs, source_masks)
        else:
            encoding_outputs, source_masks = None, None

        if self.training:
            label_smooth = self.args.label_smooth
        else:
            label_smooth = 0.0

        if mode == 'train':
            target_inputs_original = batch.trg_n_original

            # insert_word, insert_location, insert_weights = self.generate_training_labels(
            #     target_inputs_original,
            #     target_original,
            #     target_inputs,
            #     tau=1,
            #     get_delete_labels=False)

            insert_labels, terminal_labels = self.generate_training_labels(
                target_inputs_original,
                target_original,
                target_inputs,
                tau=1,
                get_delete_labels=False)

            decoding_inputs = self.io_dec.i(
                target_inputs, pos=(not self.relative_pos))
            decoding_outputs = self.decoder(decoding_inputs, target_masks,
                                            encoding_outputs, source_masks)[-1]
            slots_outputs_contents, slots_outputs_location, slots_masks, slots_targets = self.prepare_content_location_representations(
                decoding_outputs, target_masks)

            # content_slot_loss = {}
            # logp = -log_softmax(self.io_dec.o(slots_outputs_contents)).view(
            #     batch_size, -1)  # batch x len x vocab
            # insert_index = insert_word + insert_location * vocab_size

            # content_slot_loss['SLT'] = (
            #     logp.gather(1, insert_index) *
            #     insert_weights).sum() / insert_weights.sum()

            content_slot_loss = self.io_dec.sent_cost(
                insert_labels,
                slots_masks,
                slots_outputs_contents,
                label_smooth=label_smooth,
                name='SLT',
                sentence_wise=False)

            location_slot_loss = self.pointer.sent_cost(
                slots_targets[:, None, :],
                target_masks[:, 1:],
                slots_outputs_location,
                self.prepare_position_query_vectors(batch_size),
                name='PNT',
                sentence_wise=False)

            info = self.merge_losses(info,
                                     [content_slot_loss, location_slot_loss])

        elif mode == 'decoding':
            if self.args.beam_size > 1:
                raise NotImplementedError(
                    'No beam-search is available for IT-SLOT')

            if self.args.parallel_decoding:
                translation_outputs = self.parallel_decoding(
                    encoding_outputs,
                    source_masks,
                    field=dataflow[1],
                    beta=self.args.termination_penalty,
                    return_all_out=False)
            else:
                translation_outputs = self.greedy_decoding(
                    encoding_outputs,
                    source_masks,
                    field=dataflow[1],
                    beta=self.args.termination_penalty,
                    return_all_out=False)

            translation_outputs = self.fields[dataflow[1]].reverse(
                translation_outputs)

            info['src'] = source_original
            info['trg'] = target_original
            info['dec'] = translation_outputs

        return info

    def greedy_decoding(self,
                        encoding=None,
                        mask_src=None,
                        T=None,
                        field='trg',
                        beta=0,
                        return_all_out=False):
        """
        beta: termination (EoS) penalty
        """
        encoding = self.decoder.prepare_encoder(encoding)
        if T is None:
            T = encoding[0].size()[1]
        B, C = encoding[0].size()[0], encoding[0].size()[-1]
        T *= self.length_ratio
        V = len(self.fields[field].vocab)

        position_query_vectors = self.prepare_position_query_vectors(B)
        index_t = self.index_t

        # place-holders
        outs_location = torch.arange(T + 2, device=encoding[0].get_device())
        outs_location = outs_location[None, :].expand(
            B, T + 2).contiguous().float()
        outs_contents = encoding[0].new_zeros(B, T + 2).long()
        outs_contents[:, 0] = self.fields[field].vocab.stoi[
            self.fields[field].init_token]
        outs_contents[:, 1] = self.fields[field].vocab.stoi[
            self.fields[field].eos_token]
        hiddens = [
            encoding[0].new_zeros(B, T + 2, C)
            for l in range(len(self.decoder.layers) + 1)
        ]
        eos_yet = encoding[0].new_zeros(B).byte()
        all_out = []

        for t in range(T):

            # add dropout, etc.
            hiddens[0][:, :t + 2] = self.decoder.prepare_embedding(
                self.io_dec.i(outs_contents[:, :t + 2], pos=True))

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :t + 2]
                x = self.decoder.layers[l].selfattn(x, x, x)
                hiddens[l + 1][:, :t + 2] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l],
                                                     encoding[l], mask_src))

            # content, position prediction
            slot_contents, slot_location = self.prepare_content_location_representations(
                hiddens[-1][:, :t + 2])

            pos_logp = log_softmax(
                self.pointer(position_query_vectors, slot_location)).transpose(
                    2, 1).contiguous()
            con_logp = log_softmax(self.io_dec.o(slot_contents))
            if beta > 0:
                con_logp[:, :, index_t] = con_logp[:, :, index_t] - beta

            scores = pos_logp + con_logp

            # check if ended with <T> at everywhere
            eos_yet = eos_yet | (
                (scores.max(2)[1] == index_t).long().prod(1) == 1)
            if eos_yet.all():
                break

            # set scores for <T> to -INF
            scores[:, :, index_t] = -INF
            preds = scores.view(B, -1).max(1)[1]  # max again, no <T> any more
            preds_contents = preds % V
            preds_location = (preds // V).float() + 0.5

            # if sentence is ended, set to <PAD>
            preds_contents[eos_yet] = index_t
            preds_location[eos_yet] = INF  # make sure to the end.

            # reorder the tokens
            outs_contents[:, :t + 3] = torch.gather(
                torch.cat([outs_contents[:, :t + 2], preds_contents[:, None]],
                          1), 1,
                torch.cat([outs_location[:, :t + 2], preds_location[:, None]],
                          1).sort(1)[1])

            if return_all_out:
                all_out.append(outs_contents[:, :t + 3].clone())

        if return_all_out:
            return outs_contents[:, :t + 3], all_out
        return outs_contents[:, :t + 3]

    def parallel_decoding(self,
                          encoding=None,
                          mask_src=None,
                          T=None,
                          field='trg',
                          beta=0,
                          return_all_out=False):
        """
        beta: termination (EoS) penalty
        """
        encoding = self.decoder.prepare_encoder(encoding)
        if T is None:
            T = encoding[0].size()[1]
        B, C = encoding[0].size()[0], encoding[0].size()[-1]
        T *= self.length_ratio

        position_query_vectors = self.prepare_position_query_vectors(B)
        index_t = self.index_t

        # place-holders
        outs_location = torch.arange(T + 2, device=encoding[0].get_device())
        outs_location = outs_location[None, :].expand(
            B, T + 2).contiguous().float()

        outs_contents = encoding[0].new_zeros(B, T + 2).long().fill_(
            self.fields[field].vocab.stoi[self.fields[field].pad_token])
        outs_contents[:, 0] = self.fields[field].vocab.stoi[
            self.fields[field].init_token]
        outs_contents[:, 1] = self.fields[field].vocab.stoi[
            self.fields[field].eos_token]

        outs_masks = encoding[0].new_zeros(B, T + 2).float()
        outs_masks[:, :2] = 1

        hiddens = [
            encoding[0].new_zeros(B, T + 2, C)
            for l in range(len(self.decoder.layers) + 1)
        ]
        eos_yet = encoding[0].new_zeros(B).byte()
        all_out = []

        t = 0
        while True:
            hiddens[0][:, :t + 2] = self.decoder.prepare_embedding(
                self.io_dec.i(outs_contents[:, :t + 2], pos=True))

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :t + 2]
                x = self.decoder.layers[l].selfattn(x, x, x,
                                                    outs_masks[:, :t + 2])
                hiddens[l + 1][:, :t + 2] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l],
                                                     encoding[l], mask_src))

            # content, position prediction
            slot_contents, slot_location = self.prepare_content_location_representations(
                hiddens[-1][:, :t + 2])

            # for slot-based model, parallel decoding is on all positions.
            # pos_logp = log_softmax(
            #     self.pointer(
            #         position_query_vectors,
            #         slot_location,
            #         mask=outs_masks[:, 1:t + 2])).transpose(2, 1).contiguous()
            con_logp = log_softmax(self.io_dec.o(
                slot_contents)) - INF * (1 - outs_masks[:, 1:t + 2, None])

            if beta > 0:
                con_logp[:, :, index_t] = con_logp[:, :, index_t] - beta
            scores = con_logp

            # scores = pos_logp + con_logp

            # parallel decoding::
            # decoding at every posiiton and check if they are <T>
            preds_contents = scores.max(2)[1]
            is_terminated = (preds_contents == index_t)

            # check if ended with <T> at everywhere
            eos_yet = eos_yet | (is_terminated.long().prod(1) == 1)
            if eos_yet.all():
                break

            preds_location = outs_location[:, :t + 1] + 0.5
            preds_location[is_terminated] = INF

            # if sentence is ended, set to <PAD>
            preds_contents[eos_yet] = index_t
            preds_location[eos_yet] = INF  # make sure to the end.

            # reorder the tokens
            preds_outputs = torch.gather(
                torch.cat([outs_contents[:, :t + 2], preds_contents], 1), 1,
                torch.cat([outs_location[:, :t + 2], preds_location],
                          1).sort(1)[1])

            preds_masks = (preds_outputs != index_t).float()
            preds_length = preds_masks.sum(1).long()

            # check out-of-boundaries:: (force ending)
            eos_yet = eos_yet | (preds_length > (T + 1))
            if eos_yet.all():
                break

            # check the padded words. update the masks
            t = min(preds_length.max(0)[0].item() - 2, T)
            outs_masks[:, :t + 2] = preds_masks[:, :t + 2]
            outs_contents[:, :t + 2] = preds_outputs[:, :t + 2]

            if return_all_out:
                all_out.append(outs_contents[:, :t + 2].clone())

        if return_all_out:
            return outs_contents[:, :t + 2], all_out
        return outs_contents[:, :t + 2]


@register_new_model('InsertionTransformer_S')
class InsertionTransformer_S(InsertionTransformer):
    """
    InsertionTransformer with sequence-termination
    """

    def prepare_content_location_representations(self,
                                                 decoding_outputs,
                                                 target_masks=None,
                                                 terminal_labels=None):

        slots_outputs = torch.cat(
            [decoding_outputs[:, :-1], decoding_outputs[:, 1:]], 2)
        slots_outputs_contents = self.trans_c(slots_outputs)
        slots_outputs_location = self.trans_l(slots_outputs)

        if target_masks is None:
            return slots_outputs_contents, slots_outputs_location
        is_terminated = (terminal_labels.sum(-1, keepdim=True) == 0).float()

        # if terminated, then output <pad>
        slots_masks = target_masks[:, 1:] * terminal_labels * (
            1 - is_terminated) + target_masks[:, 1:] * is_terminated

        slots_targets = slots_masks / slots_masks.sum(-1, keepdim=True)
        return slots_outputs_contents, slots_outputs_location, slots_masks, slots_targets

    def forward(self, batch, mode='train', dataflow=['src', 'trg'], step=None):

        if not (dataflow[0] == 'src' and dataflow[1] == 'trg'):
            raise NotImplementedError(
                'currently in default flow from source to target.')

        info = defaultdict(lambda: 0)
        source_inputs, source_outputs, source_masks, source_original, \
            target_inputs, target_outputs, target_masks, target_original = self.prepare_data(batch, dataflow=dataflow)
        batch_size = source_inputs.size(0)

        info['sents'] = (target_outputs[:, 0] * 0 + 1).sum()
        info['tokens'] = (target_masks != 0).sum()
        info['full_tokens'] = (target_outputs != self.index_t).sum()

        if dataflow[0] is not None:
            encoding_inputs = self.io_enc.i(
                source_inputs, pos=(not self.relative_pos))
            encoding_outputs = self.encoder(encoding_inputs, source_masks)
        else:
            encoding_outputs, source_masks = None, None

        if self.training:
            label_smooth = self.args.label_smooth
        else:
            label_smooth = 0.0

        if mode == 'train':
            target_inputs_original = batch.trg_n_original
            insert_labels, terminal_labels = self.generate_training_labels(
                target_inputs_original,
                target_original,
                target_inputs,
                tau=1,
                get_delete_labels=False)
            terminal_labels = terminal_labels * target_masks[:, 1:]

            decoding_inputs = self.io_dec.i(
                target_inputs, pos=(not self.relative_pos))
            decoding_outputs = self.decoder(decoding_inputs, target_masks,
                                            encoding_outputs, source_masks)[-1]
            slots_outputs_contents, slots_outputs_location, slots_masks, slots_targets = \
                self.prepare_content_location_representations(decoding_outputs, target_masks, terminal_labels)

            content_slot_loss = self.io_dec.sent_cost(
                insert_labels,
                slots_masks,
                slots_outputs_contents,
                label_smooth=label_smooth,
                name='SLT',
                sentence_wise=False)

            location_slot_loss = self.pointer.sent_cost(
                slots_targets[:, None, :],
                target_masks[:, 1:],
                slots_outputs_location,
                self.prepare_position_query_vectors(batch_size),
                name='PNT',
                sentence_wise=False)

            info = self.merge_losses(info,
                                     [content_slot_loss, location_slot_loss])

        elif mode == 'decoding':
            if self.args.beam_size > 1:
                raise NotImplementedError(
                    'No beam-search is available for IT-SLOT')

            if self.args.parallel_decoding:
                raise NotImplementedError(
                    'The original paper does not permit parallel decoding.')
            else:
                translation_outputs = self.greedy_decoding(
                    encoding_outputs,
                    source_masks,
                    field=dataflow[1],
                    beta=self.args.termination_penalty,
                    return_all_out=False)

            translation_outputs = self.fields[dataflow[1]].reverse(
                translation_outputs)

            info['src'] = source_original
            info['trg'] = target_original
            info['dec'] = translation_outputs

        return info

    def greedy_decoding(self,
                        encoding=None,
                        mask_src=None,
                        T=None,
                        field='trg',
                        beta=0,
                        return_all_out=False):
        """
        beta: termination (EoS) penalty
        """
        encoding = self.decoder.prepare_encoder(encoding)
        if T is None:
            T = encoding[0].size()[1]
        B, C = encoding[0].size()[0], encoding[0].size()[-1]
        T *= self.length_ratio
        V = len(self.fields[field].vocab)

        position_query_vectors = self.prepare_position_query_vectors(B)
        index_t = self.index_t

        # place-holders
        outs_location = torch.arange(T + 2, device=encoding[0].get_device())
        outs_location = outs_location[None, :].expand(
            B, T + 2).contiguous().float()
        outs_contents = encoding[0].new_zeros(B, T + 2).long()
        outs_contents[:, 0] = self.fields[field].vocab.stoi[
            self.fields[field].init_token]
        outs_contents[:, 1] = self.fields[field].vocab.stoi[
            self.fields[field].eos_token]
        hiddens = [
            encoding[0].new_zeros(B, T + 2, C)
            for l in range(len(self.decoder.layers) + 1)
        ]
        eos_yet = encoding[0].new_zeros(B).byte()
        all_out = []

        for t in range(T):
            # add dropout, etc.
            hiddens[0][:, :t + 2] = self.decoder.prepare_embedding(
                self.io_dec.i(outs_contents[:, :t + 2], pos=True))

            for l in range(len(self.decoder.layers)):
                x = hiddens[l][:, :t + 2]
                x = self.decoder.layers[l].selfattn(x, x, x)
                hiddens[l + 1][:, :t + 2] = self.decoder.layers[l].feedforward(
                    self.decoder.layers[l].crossattn(x, encoding[l],
                                                     encoding[l], mask_src))

            # content, position prediction
            slot_contents, slot_location = self.prepare_content_location_representations(
                hiddens[-1][:, :t + 2])

            pos_logp = log_softmax(
                self.pointer(position_query_vectors, slot_location)).transpose(
                    2, 1).contiguous()
            con_logp = log_softmax(self.io_dec.o(slot_contents))
            if beta > 0:
                con_logp[:, :, index_t] = con_logp[:, :, index_t] - beta

            scores = pos_logp + con_logp
            preds = scores.view(B, -1).max(1)[1]  # max again, no <T> any more
            preds_contents = preds % V
            preds_location = (preds // V).float() + 0.5

            # check if ended with <T> at any place (selected EOS)
            eos_yet = eos_yet | (preds_contents == index_t)
            if eos_yet.all():
                break

            # if sentence is ended, set to <PAD>
            preds_contents[eos_yet] = index_t
            preds_location[eos_yet] = INF  # make sure to the end.

            # reorder the tokens
            outs_contents[:, :t + 3] = torch.gather(
                torch.cat([outs_contents[:, :t + 2], preds_contents[:, None]],
                          1), 1,
                torch.cat([outs_location[:, :t + 2], preds_location[:, None]],
                          1).sort(1)[1])

            if return_all_out:
                all_out.append(outs_contents[:, :t + 3].clone())

        if return_all_out:
            return outs_contents[:, :t + 3], all_out
        return outs_contents[:, :t + 3]
