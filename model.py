import torch
import math

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

    def __init__(self, d_key, drop_ratio, causal, noisy=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.noisy  = noisy
        self.p_attn = None

    def forward(self, query, key, value=None, mask=None, beta=0, tau=1):
        dot_products = matmul(query, key.transpose(1, 2))   # batch x trg_len x trg_len

        if query.dim() == 3 and self.causal: # and (query.size(1) == key.size(1)):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            tri = tri[-query.size(1):]       # caual attention may work on non-square attention.
            
            dot_products.data.sub_(tri.unsqueeze(0))

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

    def __init__(self, d_key, d_value, n_heads, drop_ratio=0.1, causal=False, noisy=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, noisy=noisy)
        self.wq = Linear(d_key,   d_key, bias=True)
        self.wk = Linear(d_key,   d_key, bias=True)
        self.wv = Linear(d_value, d_value, bias=True)
        self.wo = Linear(d_value, d_key, bias=True)
        self.n_heads = n_heads

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

        outputs = self.attention(query, key, value, mask, beta, tau)  # (B x n) x T x (D/n)
        outputs = outputs.contiguous().view(B, N, -1, D//N).transpose(2, 1).contiguous().view(B, -1, D)
        return self.wo(outputs)


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

class EncoderLayer(nn.Module):

    def __init__(self, args, causal=False, order='tdan'):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead2(
                args.d_model, args.d_model, args.n_heads,
                args.drop_ratio, causal),
            args.d_model, args.drop_ratio, order=order)
        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden, args.drop_ratio),
            args.d_model, args.drop_ratio, order=order)

    def forward(self, x, mask=None):
        return self.feedforward(self.selfattn(x, x, x, mask))


class DecoderLayer(nn.Module):

    def __init__(self, args, causal=True, noisy=False, order='tdan'):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, causal),
            args.d_model, args.drop_ratio, order=order)

        self.crossattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, noisy=noisy),  # only noisy when doing cross-attention
            args.d_model, args.drop_ratio, order=order)

        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden, args.drop_ratio),
            args.d_model, args.drop_ratio, order=order)

    def forward(self, x, encoding, p=None, mask_src=None, mask_trg=None):

        x = self.selfattn(x, x, x, mask_trg)   
        x = self.feedforward(self.crossattn(x, encoding, encoding, mask_src))
        return x


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
            x += positional_encodings_like(x)
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

    def reverse(self, outputs, **kwargs):
        return self.field.reverse(outputs.data, **kwargs)


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
                acceptance, new_masks = self.viterbi(scores, shifted_masks, self.args.constant_penalty, random=self.args.random_path)

            # visualize the paths
            if self.printer_param[1] % self.printer_param[0] == 1:
                print('rank{} sample:\t'.format(self.args.local_rank), 
                    colored_seq(self.field.reverse(targets)[0], acceptance[0, :].cpu().tolist()))

            # use another predictor to predict the beam-searched sequence!
            predictions = self.predict(outputs)
            acceptance, predictions = with_mask(acceptance, predictions, masks.byte())
            loss['ACC'] = F.cross_entropy(predictions, acceptance)
            loss['#SPEEDUP'] = 1 / (1 - acceptance.float().mean() + 1e-9)

            if self.dyn == 1:
                shifted_targets, block_outputs = with_mask(shifted_targets, block_outputs, new_masks.byte())
                loss[name] = cross_entropy_with_smooth(self.out(block_outputs), shifted_targets, label_smooth)
            else:
                new_masks = new_masks[shifted_masks.byte()]
                shifted_targets, block_outputs = with_mask(shifted_targets, block_outputs, shifted_masks.byte())
                loss[name] = cross_entropy_with_smooth(self.out(block_outputs), shifted_targets, label_smooth, self.dyn * new_masks + 1 - self.dyn)
        
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

    # def search(self, scores, K=8):
    #     batchsize, seqsize, blocksize = scores.size()
    #     scores = shift(scores, blocksize, right=True, value=-INF)  # right-shifting
    #     scores = torch.cat([scores, scores.new_zeros(batchsize, seqsize, 1) - INF], dim=2)  # safely guard
        
    #     decisions = scores.new_zeros(batchsize, seqsize, K).long() # all starts from 0
    #     outputs = scores[:, 0, :].gather(1, decisions[:, 0, :])
    #     outputs[:, 1:].add_(-INF)
        
    #     for t in range(1, seqsize):
    #         reject_outputs = outputs + scores[:, t, 0:1]
    #         accept_outputs = outputs + scores[:, t, :].gather(1, decisions[:, t-1, :] + 1)
            
    #         new_outputs = torch.cat([reject_outputs, accept_outputs], 1)
    #         decisions = torch.cat([decisions, decisions], 2)
    #         decisions[:, t, K:] = decisions[:, t-1, K:] + 1

    #         outputs, sorted_ind = new_outputs.topk(K, dim=1)
    #         decisions = decisions.gather(2, sorted_ind[:, None, :].expand(batchsize, seqsize, K))

    #     best_decision = decisions[:, :, 0]  
    #     acceptance = (best_decision != 0).long()
    #     new_mask = scores.new_zeros(batchsize, seqsize, blocksize).scatter_(2, best_decision.unsqueeze(-1), 1)
    #     new_mask = shift(new_mask, blocksize, right=False)

    #     return acceptance, new_mask



class Encoder(nn.Module):

    def __init__(self, field, args, causal=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(args, causal, order=args.block_order) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)
        
        if args.normalize_emb:
            self.layernorm = LayerNorm(args.d_model)
        self.field = field
        self.d_model = args.d_model
        self.share_embeddings = args.share_embeddings
        self.normalize_emb = args.normalize_emb

    def prepare_embedding(self, embedding):
        embedding = self.dropout(embedding)
        if self.normalize_emb:
            embedding = self.layernorm(embedding)
        return embedding

    def forward(self, x, mask=None):

        encoding = [x]
        x = self.prepare_embedding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            encoding.append(x)

        return encoding


class Decoder(nn.Module):

    def __init__(self, field, args, causal=True, noisy=False):

        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(args, causal, noisy, order=args.block_order)
            for i in range(args.n_layers)])

        self.dropout = nn.Dropout(args.drop_ratio)

        if args.normalize_emb:
            self.layernorm = LayerNorm(args.d_model)

        self.d_model = args.d_model
        self.field = field
        self.length_ratio = args.length_ratio
        self.cross_attn_fashion = args.cross_attn_fashion
        self.normalize_emb = args.normalize_emb

    def prepare_encoder(self, encoding):
        if self.cross_attn_fashion == 'reverse':
            encoding = encoding[1:][::-1]
        elif self.cross_attn_fashion == 'last_layer':
            encoding = [encoding[-1] for _ in range(len(self.layers))]
        else:
            encoding = encoding[1:]
        return encoding

    def prepare_embedding(self, embedding):
        embedding = self.dropout(embedding)
        if self.normalize_emb:
            embedding = self.layernorm(embedding)
        return embedding

    def forward(self, x, encoding=None, mask_src=None, mask_trg=None):

        x = self.dropout(x)
        if self.normalize_emb:
            x = self.layernorm(x)
            
        encoding = self.prepare_encoder(encoding)

        for l, (layer, enc) in enumerate(zip(self.layers, encoding)):
            x = layer(x, enc, mask_src=mask_src, mask_trg=mask_trg)
        return x

    def greedy(self, io_dec, encoding, mask_src=None, mask_trg=None):

        encoding = self.prepare_encoder(encoding)
        B, T, C = encoding[0].size()  # batch_size, decoding-length, size
        T *= self.length_ratio

        outs = Variable(encoding[0].data.new(B, T + 1).long().fill_(
                    self.field.vocab.stoi['<init>']))
        hiddens = [Variable(encoding[0].data.new(B, T, C).zero_())
                    for l in range(len(self.layers) + 1)]
        # embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].data.new(B).byte().zero_()

        for t in range(T):
            
            # add dropout, etc.
            hiddens[0][:, t] = self.prepare_embedding(hiddens[0][:, t] + io_dec.i(outs[:, t], pos=False))

            for l in range(len(self.layers)):
                x = hiddens[l][:, :t+1]
                x = self.layers[l].selfattn(hiddens[l][:, t:t+1], x, x)   # we need to make the dimension 3D
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].crossattn(x, encoding[l], encoding[l], mask_src))[:, 0]

            _, preds = io_dec.o(hiddens[-1][:, t]).max(-1)
            preds[eos_yet] = self.field.vocab.stoi['<pad>']

            eos_yet = eos_yet | (preds.data == self.field.vocab.stoi['<eos>'])
            outs[:, t + 1] = preds
            if eos_yet.all():
                break

        return outs[:, 1:t+2]

    def beam_search(self, io_dec, encoding, mask_src=None, mask_trg=None, width=2, alpha=0.6):  # width: beamsize, alpha: length-norm
        
        encoding = self.prepare_encoder(encoding)

        W = width
        B, T, C = encoding[0].size()

        # expanding
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(B, W, T, C).contiguous().view(B * W, T, C)
        mask_src = mask_src[:, None, :].expand(B, W, T).contiguous().view(B * W, T)

        T *= self.length_ratio
        outs = Variable(encoding[0].data.new(B, W, T + 1).long().fill_(
            self.field.vocab.stoi['<pad>']))
        outs[:, :, 0] = self.field.vocab.stoi['<init>']

        logps = Variable(encoding[0].data.new(B, W).float().fill_(0))  # scores
        hiddens = [Variable(encoding[0].data.new(B, W, T, C).zero_())  # decoder states: batch x beamsize x len x h
                    for l in range(len(self.layers) + 1)]

        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].data.new(B, W).byte().zero_()  # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(INF)[:, :, None].expand(B, W, W).contiguous()  # --- BUG, logps < 0 assign INF here 
                                                                                        # --- UPDATE: Aug 9, 2018: BUG again, expand needs contiguous
                                                                                        # --- otherwise everything will become 0.
        eos_mask[:, :, 0] = 0  # batch x beam x beam

        for t in range(T):
            hiddens[0][:, :, t] = self.prepare_embedding(hiddens[0][:, :, t] + io_dec.i(outs[:, :, t], pos=False))

            for l in range(len(self.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.layers[l].selfattn(x[:, -1:, :], x, x)
                hiddens[l + 1][:, :, t] = self.layers[l].feedforward(
                    self.layers[l].crossattn(x, encoding[l], encoding[l], mask_src)).view(
                        B, W, C)

            # topk2_logps: scores, topk2_inds: top word index at each beam, batch x beam x beam
            topk2_logps = log_softmax(io_dec.o(hiddens[-1][:, :, t]))
            topk2_logps[:, :, self.field.vocab.stoi['<pad>']] = -INF
            topk2_logps, topk2_inds = topk2_logps.topk(W, dim=-1)

            # mask out the sentences which are finished
            topk2_logps = topk2_logps * Variable(eos_yet[:, :, None].float() * eos_mask + 1 - eos_yet[:, :, None].float())
            topk2_logps = topk2_logps + logps[:, :, None]

            if t == 0:
                logps, topk_inds = topk2_logps[:, 0].topk(W, dim=-1)
            else:
                logps, topk_inds = topk2_logps.view(B, W * W).topk(W, dim=-1)

            topk_beam_inds = topk_inds.div(W)
            topk_token_inds = topk2_inds.view(B, W * W).gather(1, topk_inds)
            eos_yet = eos_yet.gather(1, topk_beam_inds.data)
            
            # logps = logps * (1 - Variable(eos_yet.float()) * 1 / (t + 2)).pow(alpha) # -- bug
            logps = logps * (1 + Variable(eos_yet.float()) * 1 / (t + 1)).pow(alpha)
            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs)).contiguous()
            outs[:, :, t + 1] = topk_token_inds
            topk_beam_inds = topk_beam_inds[:, :, None, None].expand_as(hiddens[0]).contiguous()

            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds)
            eos_yet = eos_yet | (topk_token_inds.data == self.field.vocab.stoi['<eos>'])
            if eos_yet.all():
                return outs[:, 0, 1:]
        return outs[:, 0, 1:]


class Transformer(nn.Module):

    def __init__(self, src, trg, args):
        super().__init__()
        self.encoder = Encoder(src, args, causal=args.causal_enc)
        self.decoder = Decoder(trg, args, causal=True)
        
        if args.multi_width > 1:
            self.io_dec = MulIO(trg, args)
            self.io_enc = IO(src, args)
        else:
            self.io_dec = IO(trg, args)
            self.io_enc = IO(src, args)

        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

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

    def encoding(self, encoder_inputs, encoder_masks):
        return self.encoder(self.io_enc.i(encoder_inputs, pos=True), encoder_masks)

    def decoding(self, encoding_outputs, encoder_masks, decoder_inputs, decoder_masks,
                decoding=False, beam=1, alpha=0.6, return_probs=False):

        if (return_probs and decoding) or (not decoding):
            out = self.decoder(self.io_dec.i(decoder_inputs, pos=True), encoding_outputs, encoder_masks, decoder_masks)

        if decoding:
            if beam == 1:  # greedy decoding
                output = self.decoder.greedy(self.io_dec, encoding_outputs, encoder_masks, decoder_masks)
            else:
                output = self.decoder.beam_search(self.io_dec, encoding_outputs, encoder_masks, decoder_masks, beam, alpha)

            if return_probs:
                return output, out, softmax(self.io_dec.o(out))
            return output

        if return_probs:
            return out, softmax(self.io_dec.o(out))
        return out


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
        encoding_outputs = self.encoding(source_inputs, source_masks)

        if not decoding:

            # Maximum Likelihood Training (with label smoothing trick)
            decoding_outputs = self.decoding(encoding_outputs, source_masks, target_inputs, target_masks)        
            loss = self.io_dec.cost(target_outputs, target_masks, outputs=decoding_outputs, label_smooth=self.args.label_smooth)
            
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
            if self.args.multi_width > 1: # -- the newly introduced block-wise decoding -- 
                decoding_outputs = self.blockwise_parallel_decoding(encoding_outputs, source_masks)
            else:
                decoding_outputs = self.decoding(encoding_outputs, source_masks, target_inputs, target_masks, decoding=True, return_probs=False)

            if reverse:
                source_outputs = self.io_enc.reverse(source_outputs)
                target_outputs = self.io_dec.reverse(target_outputs)
                decoding_outputs, saved_time, pred_acc, decisions = self.io_dec.reverse(decoding_outputs, width=self.args.multi_width, return_saved_time=True)
                
                info['saved_time'] = saved_time
                info['pred_acc'] = pred_acc
                info['decisions'] = decisions

            info['src'] = source_outputs
            info['trg'] = target_outputs
            info['dec'] = decoding_outputs
        
        return info


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
                    # dynamic block-wise based on my prediction.
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