import torch
import math
import copy

from collections import defaultdict
from abc import ABCMeta, abstractmethod
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import computeGLEU, masked_sort, unsorted, colored_seq

INF = 1e10
TINY = 1e-9

# -- -- helper functions ----- #
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def positional_encodings_like(x, t=None):   # hope to be differentiable
    if t is None:
        positions = torch.arange(0, x.size(-2)) # .expand(*x.size()[:2])
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
        positions = Variable(positions)
    else:
        positions = t
    positions = positions.float()

    # channels
    channels = torch.arange(0, x.size(-1), 2).float() / x.size(-1) # 0 2 4 6 ... (256)
    if x.is_cuda:
        channels = channels.cuda(x.get_device())
    channels = 1 / (10000 ** Variable(channels))

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x).contiguous()

    return encodings

def linear_wn(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def cosine_sim(x, y):
    x = x / (x.norm(dim=-1, keepdim=True).expand_as(x) + TINY)
    y = y / (y.norm(dim=-1, keepdim=True).expand_as(y) + TINY)
    return (x * y).sum(dim=-1)

def with_mask(targets, out, input_mask=None, return_mask=False):
    if input_mask is None:
        input_mask = (targets != 1)
    
    out_mask = input_mask.unsqueeze(-1).expand_as(out)

    if return_mask:
        return targets[input_mask], out[out_mask].view(-1, out.size(-1)), the_mask
    return targets[input_mask], out[out_mask].view(-1, out.size(-1))

def demask(inputs, the_mask):
    # inputs: 1-D sequences
    # the_mask: batch x max-len
    outputs = Variable((the_mask == 0).long().view(-1))  # 1-D
    indices = torch.arange(0, outputs.size(0))
    if inputs.is_cuda:
        indices = indices.cuda(inputs.get_device())
    indices = indices.view(*the_mask.size()).long()
    indices = indices[the_mask]
    outputs[indices] = inputs
    return outputs.view(*the_mask.size())

# F.softmax has strange default behavior, normalizing over dim 0 for 3D inputs
def softmax(x):
    return F.softmax(x, dim=-1)

def log_softmax(x):
    return F.log_softmax(x, dim=-1)

def logsumexp(x, dim=-1):
    x_max = x.max(dim, keepdim=True)[0]
    return torch.log(torch.exp(x - x_max.expand_as(x)).sum(dim, keepdim=True) + TINY) + x_max

def gumbel_softmax(input, beta=0.5, tau=1.0):
    noise = input.data.new(*input.size()).uniform_()
    noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
    return softmax((input + beta * Variable(noise)) / tau)

def argmax(x):  # return the one-hot vectors
    shape = x.size()
    _, ind = x.max(dim=-1)
    x_hard = Variable(x.data.new(x.size()).zero_().view(-1, shape[-1]))
    x_hard.scatter_(1, ind.view(-1, 1), 1)
    x_hard = x_hard.view(*shape)
    return x_hard

def cross_entropy_with_smooth(outputs, targets, label_smooth=0.1, reweight=None):
    logits = log_softmax(outputs)

    if reweight is None:
        return F.nll_loss(logits, targets) * (1 - label_smooth) - logits.mean() * label_smooth
    else:
        
        nll_loss = (F.nll_loss(logits, targets, reduction='none') * reweight).mean()
        return nll_loss * (1 - label_smooth) - logits.mean() * label_smooth

def shift(x, n, right = False, value=0):
    if x.dim() == 2:
        x = x.unsqueeze(-1).expand(*x.size()[:2], n)
    new_x = x.new_zeros(*x.size()) + value
    new_x[:, :, 0] = x[:, :, 0]

    for i in range(1, n):
        if not right:
            new_x[:, :-i, i] = x[:, i:, i]
        else:
            new_x[:, i:, i] = x[:, :-i, i]
    return new_x

# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)

def pad_to_match(x, y):
    x_len, y_len = x.size(1), y.size(1)
    if x_len == y_len:
        return x, y
    extra = x.data.new(x.size(0), abs(y_len - x_len)).fill_(1)
    if x_len < y_len:
        return torch.cat((x, extra), 1), y
    return x, torch.cat((y, extra), 1)

# --- Top K search with PQ (used in Non-Autoregressive NMT)
def topK_search(logits, mask_src, N=100):
    # prepare data
    nlogP = -log_softmax(logits).data
    maxL = nlogP.size(-1)
    overmask = torch.cat([mask_src[:, :, None],
                        (1 - mask_src[:, :, None]).expand(*mask_src.size(), maxL-1) * INF
                        + mask_src[:, :, None]], 2)
    nlogP = nlogP * overmask

    batch_size, src_len, L = logits.size()
    _, R = nlogP.sort(-1)

    def get_score(data, index):
        # avoid all zero
        # zero_mask = (index.sum(-2) == 0).float() * INF
        return data.gather(-1, index).sum(-2)

    heap_scores = torch.ones(batch_size, N) * INF
    heap_inx = torch.zeros(batch_size, src_len, N).long()
    heap_scores[:, :1] = get_score(nlogP, R[:, :, :1])
    if nlogP.is_cuda:
        heap_scores = heap_scores.cuda(nlogP.get_device())
        heap_inx = heap_inx.cuda(nlogP.get_device())

    def span(ins):
        inds = torch.eye(ins.size(1)).long()
        if ins.is_cuda:
            inds = inds.cuda(ins.get_device())
        return ins[:, :, None].expand(ins.size(0), ins.size(1), ins.size(1)) + inds[None, :, :]

    # iteration starts
    for k in range(1, N):
        cur_inx = heap_inx[:, :, k-1]
        I_t = span(cur_inx).clamp(0, L-1)  # B x N x N
        S_t = get_score(nlogP, R.gather(-1, I_t))
        S_t, _inx = torch.cat([heap_scores[:, k:], S_t], 1).sort(1)
        S_t[:, 1:] += ((S_t[:, 1:] - S_t[:, :-1]) == 0).float() * INF  # remove duplicates
        S_t, _inx2 = S_t.sort(1)
        I_t = torch.cat([heap_inx[:, :, k:], I_t], 2).gather(
                        2, _inx.gather(1, _inx2)[:, None, :].expand(batch_size, src_len, _inx.size(-1)))
        heap_scores[:, k:] = S_t[:, :N-k]
        heap_inx[:, :, k:] = I_t[:, :, :N-k]

    # get the searched
    output = R.gather(-1, heap_inx)
    output = output.transpose(2, 1).contiguous().view(batch_size * N, src_len)  # (B x N) x Ts
    output = Variable(output)
    mask_src = mask_src[:, None, :].expand(batch_size, N, src_len).contiguous().view(batch_size * N, src_len)

    return output, mask_src


class Linear(nn.Linear):
    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio, pos=0, order='tdan'):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos
        self.order = order

    def forward(self, *x):
        y = x    
        assert len(self.order) >= 4, 'at least 4 operations in one block'
        assert self.order[0] == 't', 'we must start from transformation'
        for c in self.order:
            if c == 't':
                y = self.layer(*y)
            elif c == 'd':
                y = self.dropout(y)
            elif c == 'a':
                y = x[self.pos] + y
            elif c == 'n':
                y = self.layernorm(y)
            else:
                raise NotImplementedError

        return y
        # return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))


class HighwayBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.gate = Linear(d_model, 1)
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        g = torch.sigmoid(self.gate(x[0])).expand_as(x[0])
        return self.layernorm(x[0] * g + self.dropout(self.layer(*x)) * (1 - g))


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal, noisy=False, local=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.local = local
        self.window = 2
        self.noisy  = noisy
        self.p_attn = None

    def forward(self, query, key, value=None, mask=None, beta=0, tau=1):
        dot_products = matmul(query, key.transpose(1, 2))   # batch x trg_len x trg_len

        if query.dim() == 3 and self.causal: # and (query.size(1) == key.size(1)):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            tri = tri[-query.size(1):]       # caual attention may work on non-square attention.
            dot_products.data.sub_(tri.unsqueeze(0))

        if self.local:
            window_mask = key.new_ones(key.size(1), key.size(1))
            window_mask = (window_mask.triu(self.window+1) + window_mask.tril(-self.window-1)) * INF
            dot_products.data.sub_(window_mask.unsqueeze(0))

        if mask is not None:
            if dot_products.dim() == 2:
                assert mask.dim() == 2, "only works on 2D masks"
                dot_products.data -= ((1 - mask) * INF)
            else:
                if mask.dim() == 2:
                    dot_products.data -= ((1 - mask[:, None, :]) * INF)
                else:
                    dot_products.data -= ((1 - mask) * INF)

        if value is None:
            return dot_products

        logits = dot_products / self.scale
        if (not self.noisy): # or (not self.training):
            probs = softmax(logits)
        else:
            probs = gumbel_softmax(logits, beta=beta, tau=tau)
        self.p_attn = probs

        # return the attention results
        return matmul(self.dropout(probs), value)


class MultiHead2(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio=0.1, causal=False, noisy=False, local=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, noisy=noisy, local=local)
        self.wq = Linear(d_key,   d_key, bias=True)
        self.wk = Linear(d_key,   d_key, bias=True)
        self.wv = Linear(d_value, d_value, bias=True)
        self.wo = Linear(d_value, d_key, bias=True)
        self.n_heads = n_heads
        self.local = local

    def forward(self, query, key, value, mask=None, beta=0, tau=1):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)   # B x T x D
        B, Tq, D = query.size()
        _, Tk, _ = key.size()
        N = self.n_heads

        # reshape query-key-value for multi-head attention
        query, key, value = (x.contiguous().view(B, -1, N, D//N).transpose(2, 1).contiguous().view(B*N, -1, D//N) for x in (query, key, value))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, :].expand(B, N, Tk).contiguous().view(B*N, -1)
            else:
                mask = mask[:, None, :, :].expand(B, N, Tq, Tk).contiguous().view(B * N, Tq, Tk)

        # local mask
        # if self.local:
        #     T = torch.arange(Tq, device=mask.get_device())[None, None, :].expand(B*N, Tq, Tk)
        #     T = (T - T.transpose(1, 2)).float()
        #     t = 2

        #     def H(a):
        #         return torch.sigmoid(10 * (a + 0.5))

        #     new_mask = ((H(T + t) - H(T - t - 1)) > 0.5).float()
        #     if mask.dim() == 2:
        #         mask = mask[:,None,:] * new_mask
        #     else:
        #         mask = mask * new_mask

        outputs = self.attention(query, key, value, mask, beta, tau)  # (B x n) x T x (D/n)
        outputs = outputs.contiguous().view(B, N, -1, D//N).transpose(2, 1).contiguous().view(B, -1, D)
        return self.wo(outputs)


class MultiHeadConv(nn.Module):

    def __init__(self, d_model, max_width=4):
        super().__init__()
        self.max_width = max_width
        self.convs = nn.ModuleList([nn.Conv1d(d_model // max_width, d_model // max_width, n, stride=1) 
                                    for n in range(1, max_width + 1)])

    def forward(self, x):
        batchsize, d_model, length = x.size()
        x = x.view(batchsize, d_model // self.max_width, self.max_width, length)
        o = torch.cat([self.convs[k](F.pad(x[:,:,k,:], (0, k))) for k in range(self.max_width)], 1)
        return o

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, drop_ratio=0.1, d_output=None):
        super().__init__()
        if d_output is None:
            d_output = d_model

        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(drop_ratio)


    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))  # adding dropout in feedforward layer
        # return self.linear2(F.relu(self.linear1(x)))


class Block(nn.Module):

    def __init__(self, args, causal=False, cross=False, order='tdan', local=False):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead2(
                args.d_model, args.d_model, args.n_heads,
                args.drop_ratio, causal, local=local),
            args.d_model, args.drop_ratio, order=order)
        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden, args.drop_ratio),
            args.d_model, args.drop_ratio, order=order)

        if cross:
            self.crossattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_cross_heads, args.drop_ratio),  
            args.d_model, args.drop_ratio, order=order)

        self.cross = cross
        self.causal = causal
    
    def forward(self, x, x_mask=None, y=None, y_mask=None):
        x = self.selfattn(x, x, x, x_mask)
        if self.cross:
            assert y is not None, 'cross attention needs source information'
            x = self.crossattn(x, y, y, y_mask)
        
        return self.feedforward(x)


class Stack(nn.Module):

    """
    --- Merge the Transformer's encoder & decoder into ONE class --
    """

    def __init__(self, field, args, causal=False, cross=False, local=0):

        super().__init__()
        
        self.layers = nn.ModuleList(
            [Block(args, causal, cross, order=args.block_order, local=(i < local))
            for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)

        if args.normalize_emb:
            self.layernorm = LayerNorm(args.d_model)

        self.field = field
        self.d_model = args.d_model
        self.share_embeddings = args.share_embeddings
        self.cross_attn_fashion = args.cross_attn_fashion
        self.normalize_emb = args.normalize_emb

    def prepare_encoder(self, encoding):
        if encoding is None:
            return encoding

        if len(encoding) > len(self.layers):
            encoding = encoding[1:]

        if self.cross_attn_fashion == 'reverse':
            encoding = encoding[::-1]
        elif self.cross_attn_fashion == 'last_layer':
            encoding = [encoding[-1] for _ in range(len(self.layers))]
        else:
            pass

        return encoding

    def prepare_embedding(self, embedding):
        embedding = self.dropout(embedding)
        if self.normalize_emb:
            embedding = self.layernorm(embedding)
        return embedding

    def forward(self, x, x_mask, encoding=None, y_mask=None):

        outputs = [x]
        x = self.prepare_embedding(x)
        encoding = self.prepare_encoder(encoding)

        for l, layer in enumerate(self.layers):
            y = encoding[l] if encoding is not None else None
            x = layer(x, x_mask, y, y_mask)
            outputs.append(x)

        return outputs


class IO(nn.Module):
    
    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.out = nn.Linear(args.d_model, len(field.vocab), bias=False)
        self.scale = math.sqrt(args.d_model)

    def i(self, x, pos=True):
        x = F.embedding(x, self.out.weight * self.scale)
        if pos:
            x = x + positional_encodings_like(x)
        return x

    def o(self, x):
        return self.out(x)

    def cost(self, targets, masks, outputs, label_smooth=0.0, name=None):
        loss = dict()
        if name is None:
            name = 'MLE'
        targets, outputs = with_mask(targets, outputs, masks.byte())
        loss[name] = cross_entropy_with_smooth(self.o(outputs), targets, label_smooth)
        return loss

    def acc(self, targets, masks, outputs):
        with torch.cuda.device_of(targets):
            targets, outputs = with_mask(targets, outputs, masks.byte())
            return (self.o(outputs).max(-1)[1] == targets).float().tolist()


class MulIO(IO):
    """
        IO of navie multi-step prediction.
        For "out" mode, it predicts multiple words using deconv.
    """
    def __init__(self, field, args):
        super().__init__(field, args)

        # TODO: experimental: just have a try?
        self.args = args
        self.width = args.multi_width
        self.dyn = args.dyn
        self.rounter = nn.Linear(args.d_model, args.d_model * self.width)
        self.predictor = FeedForward(args.d_model, args.d_hidden, d_output=2)
        self.printer_param = [50 * self.args.inter_size, 0]

    def expand(self, x):
        return self.rounter(x).view(*x.size(), self.width).contiguous()

    def o(self, x, full=False):
        x = self.expand(x)
        if not full:
            if x.dim() == 4:
                x = x[:, :, :, 0] 
            else:
                x = x[:, :, 0]
        else:
            x = x.transpose(-1, -2)
        return self.out(x)

    def cost(self, targets, masks, outputs, label_smooth=0.0, name=None):
        
        # some internal printing setup
        self.printer_param[1] += 1

        loss = dict()
        if name is None:
            name = 'MLE'

        shifted_targets, shifted_masks = shift(targets, self.width), shift(masks, self.width)
        block_outputs = self.expand(outputs).transpose(3, 2)    # batch_size x seq-size x block-size x d_model

        if self.dyn == 0:
            shifted_targets, block_outputs = with_mask(shifted_targets, block_outputs, shifted_masks.byte())
            loss[name] = cross_entropy_with_smooth(self.out(block_outputs), shifted_targets, label_smooth)

        else:   

            # -- exact search for the best latent sequence using viterbi-decoding -- #
            with torch.no_grad():
                scores = log_softmax(self.out(block_outputs)).gather(
                    -1, shifted_targets.unsqueeze(-1)).squeeze(-1)    # batch_size x seq-size x block-size
                acceptance, new_masks = self.viterbi(scores, shifted_masks, self.args.constant_penalty, random=False)
                _, new_random_masks   = self.viterbi(scores, shifted_masks, self.args.constant_penalty, random=True)

            # visualize the paths
            if self.printer_param[1] % self.printer_param[0] == 1:
                print('rank{} sample:\t'.format(self.args.local_rank), 
                    colored_seq(self.field.reverse(targets)[0], acceptance[0, :].cpu().tolist()))

            # use another predictor to predict the beam-searched sequence!
            predictions = self.predict(outputs)
            acceptance, predictions = with_mask(acceptance, predictions, masks.byte())
            loss['ACC'] = F.cross_entropy(predictions, acceptance)
            loss['#SPEEDUP'] = 1 / (1 - acceptance.float().mean() + 1e-9)

            optima_targets, optima_block_outputs = with_mask(shifted_targets, block_outputs, new_masks.byte())
            random_targets, random_block_outputs = with_mask(shifted_targets, block_outputs, new_random_masks.byte())
            loss[name]   =  cross_entropy_with_smooth(self.out(optima_block_outputs), optima_targets, label_smooth) * self.dyn + \
                            cross_entropy_with_smooth(self.out(random_block_outputs), random_targets, label_smooth) * (1 - self.dyn)
        
        return loss

    def predict(self, outputs):
        return self.predictor(outputs)

    def viterbi(self, scores, shifted_masks, c=0, random=False):
        """
        scores: loglikelihood
        c: penalty for autoregressive decoding
        """
        batchsize, seqsize, blocksize = scores.size()

        scores[:, :, 0] = scores[:, :, 0] - c
        if random:
            scores = torch.rand_like(scores)

        scores = shift(scores * shifted_masks, blocksize, right=True, value=-INF)  # right-shifting

        decisions = scores.new_zeros(batchsize, seqsize, blocksize).long() # all starts from 0
        outputs   = scores.new_zeros(batchsize, seqsize, blocksize).add_(-INF)
        outputs[:, 0, 0] = scores[:, 0, 0]

        for t in range(1, seqsize):
            
            max_outputs, max_indx = outputs[:, t-1].max(1)

            outputs[:, t, 0] = max_outputs + scores[:, t, 0]            # best score for reject
            outputs[:, t, 1:] = outputs[:, t-1, :-1] + scores[:, t, 1:] # best score for accept 1,2,3,...

            reject_decisions = decisions[:, :t].gather(2, max_indx[:, None, None].expand(batchsize, t, 1)) # best reject decision
            decisions[:,  t, 1:] = decisions[:, t-1, :-1] + 1
            decisions[:, :t, 1:] = decisions[:, :t,  :-1]
            decisions[:,  t, :1] = 0
            decisions[:, :t, :1] = reject_decisions
            
        best_outputs, best_indx = outputs[:, -1, :].max(1)
        best_decision = decisions.gather(2, best_indx[:, None, None].expand(batchsize, seqsize, 1))

        acceptance = (best_decision.squeeze(-1) != 0).long()
        new_masks = scores.new_zeros(batchsize, seqsize, blocksize).scatter_(2, best_decision, 1)
        new_masks = shift(new_masks, blocksize, right=False) * shifted_masks
        return acceptance, new_masks

    def search(self, scores, K=8):
        batchsize, seqsize, blocksize = scores.size()
        scores = shift(scores, blocksize, right=True, value=-INF)  # right-shifting
        scores = torch.cat([scores, scores.new_zeros(batchsize, seqsize, 1) - INF], dim=2)  # safely guard
        
        decisions = scores.new_zeros(batchsize, seqsize, K).long() # all starts from 0
        outputs = scores[:, 0, :].gather(1, decisions[:, 0, :])
        outputs[:, 1:].add_(-INF)
        
        for t in range(1, seqsize):
            reject_outputs = outputs + scores[:, t, 0:1]
            accept_outputs = outputs + scores[:, t, :].gather(1, decisions[:, t-1, :] + 1)
            
            new_outputs = torch.cat([reject_outputs, accept_outputs], 1)
            decisions = torch.cat([decisions, decisions], 2)
            decisions[:, t, K:] = decisions[:, t-1, K:] + 1

            outputs, sorted_ind = new_outputs.topk(K, dim=1)
            decisions = decisions.gather(2, sorted_ind[:, None, :].expand(batchsize, seqsize, K))

        best_decision = decisions[:, :, 0]  
        acceptance = (best_decision != 0).long()
        new_mask = scores.new_zeros(batchsize, seqsize, blocksize).scatter_(2, best_decision.unsqueeze(-1), 1)
        new_mask = shift(new_mask, blocksize, right=False)

        return acceptance, new_mask


class Seq2Seq(nn.Module):
    """
    somehow an abstract class for seq2seq models.
    provide basic input preprocessing parts
    """
    def trainable_parameters(self):
        param = [p for p in self.parameters() if p.requires_grad] 
        return [param]

    def prepare_masks(self, inputs):
        field, text = inputs
        if text.ndimension() == 2:  # index inputs
            masks = (text.data != self.fields[field].vocab.stoi['<pad>']).float()
        else:                       # one-hot vector inputs
            masks = (text.data[:, :, self.fields[field].vocab.stoi['<pad>']] != 1).float()
        return masks

    def prepare_field(self, batch, field):

        if len(field.split('_')) == 2: # injecting noise
            field, noise_level = field.split('_')
            data = batch.dataset.fields[field].reapply_noise(getattr(batch, field), noise_level)
        else:
            data = getattr(batch, field)
            
        inputs = data[:, :-1].contiguous()
        outputs = data[:, 1:].contiguous()
        masks = self.prepare_masks((field, outputs))
        return inputs, outputs, masks
        
    def prepare_data(self, batch, dataflow=['src', 'trg'], noise=None):
        # get the data
        data = dict()
        _dataflow = copy.deepcopy(dataflow)
        if noise is not None:
            _dataflow[0] = _dataflow[0] + '_' + noise

        for i, v in enumerate(_dataflow):
            if v not in data:
                data[v] = self.prepare_field(batch, v)
        output = [x for v in _dataflow for x in data[v]]
        return tuple(output)

