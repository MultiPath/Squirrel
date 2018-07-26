import torch
import math

from abc import ABCMeta, abstractmethod
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import computeGLEU, masked_sort, unsorted

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
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    channels = torch.arange(0, x.size(-1), 2) / x.size(-1) # 0 2 4 6 ... (256)
    if x.is_cuda:
        channels = channels.cuda(x.get_device())
    channels = 1 / (10000 ** Variable(channels))

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)

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

def mask(targets, out, input_mask=None, return_mask=False):
    if input_mask is None:
        input_mask = (targets != 1)
    out_mask = input_mask.unsqueeze(-1).expand_as(out)

    if return_mask:
        return targets[input_mask], out[out_mask].view(-1, out.size(-1)), the_mask
    return targets[input_mask], out[out_mask].view(-1, out.size(-1))

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

def shift(x, n = 3):
    xs = [x.unsqueeze(-1)]
    for i in range(1, n+1):
        p = x.new_zeros(*x.size())
        p[:, :-i] = x[:, i:]
        xs.append(p.unsqueeze(-1))
    return torch.cat(xs, -1)

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

    def __init__(self, layer, d_model, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))
        ## optional: add layernorm inside

class HighwayBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.gate = Linear(d_model, 1)
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        g = F.sigmoid(self.gate(x[0])).expand_as(x[0])
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

        if query.dim() == 3 and self.causal and (query.size(1) == key.size(1)):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
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
            mask = mask[:, None, :].expand(B, N, Tk).contiguous().view(B*N, -1)

        outputs = self.attention(query, key, value, mask, beta, tau)  # (B x n) x T x (D/n)
        outputs = outputs.contiguous().view(B, N, -1, D//N).transpose(2, 1).contiguous().view(B, -1, D)
        return self.wo(outputs)


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, drop_ratio=0.1):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(drop_ratio)


    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))  # adding dropout in feedforward layer
        # return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, args, causal=False):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead2(
                args.d_model, args.d_model, args.n_heads,
                args.drop_ratio, causal),
            args.d_model, args.drop_ratio)
        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden, args.drop_ratio),
            args.d_model, args.drop_ratio)

    def forward(self, x, mask=None):
        return self.feedforward(self.selfattn(x, x, x, mask))


class DecoderLayer(nn.Module):

    def __init__(self, args, causal=True, positional=False, noisy=False):
        super().__init__()
        self.positional = positional
        self.selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, causal),
            args.d_model, args.drop_ratio)

        self.crossattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, noisy=noisy),  # only noisy when doing cross-attention
            args.d_model, args.drop_ratio)

        if positional:
            self.pos_selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,   # positional attention (optional)
                    args.drop_ratio, causal),
            args.d_model, args.drop_ratio, pos=2)

        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden, args.drop_ratio),
            args.d_model, args.drop_ratio)

    def forward(self, x, encoding, p=None, mask_src=None, mask_trg=None):

        x = self.selfattn(x, x, x, mask_trg)   #
        if self.positional:
            pos_encoding = positional_encodings_like(x)
            x = self.pos_selfattn(pos_encoding, pos_encoding, x, mask_trg)  # positional attention
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

    def cost(self, targets, masks, outputs, label_smooth=0.0):
        targets, outputs = with_mask(targets, outputs, masks.byte())
        logits = log_softmax(self.o(outputs))
        return F.nll_loss(logits, targets) * (1 - label_smooth) - logits.mean() * label_smooth

    def acc(self, targets, masks, outputs):
        with torch.cuda.device_of(targets):
            targets, outputs = with_mask(targets, outputs, masks.byte())
            return (self.o(outputs).max(-1)[1] == targets).float().tolist()

    def reverse(self, outputs):
        return self.field.reverse(outputs.data, self.args.char)


class PryIO(IO):
    """
        IO of navie multi-step prediction.
        For "out" mode, it predicts multiple words using deconv.
    """
    def __init__(self, field, args):
        super().__init__(field, args)

        # TODO: experimental: just have a try?
        self.deconv1 = nn.ConvTranspose1d(args.d_model, args.d_model, 2, 2)
    
    def expand(self, x):
        S = x.size()
        x = x.view(-1, S[-1]).contiguous().unsqueeze(-1)
        x = self.deconv1(x)
        # x = self.deconv1(F.relu(x))
        return x.view(*S, -1).contiguous()

    def o(self, x):
        x = self.expand(x)
        if x.dim() == 4:
            x = x[:, :, :, 0] 
        else:
            x = x[:, :, 0]

        return self.out(x)

    def cost(self, targets, masks, outputs, label_smooth=0.0):
        targets, masks = shift(targets, 1), shift(masks, 1)
        outputs = self.expand(outputs).transpose(3, 2)

        # print(targets.size(), masks.size(), outputs.size())
        targets, outputs = with_mask(targets, outputs, masks.byte())
        logits = log_softmax(self.out(outputs))
        return F.nll_loss(logits, targets) * (1 - label_smooth) - logits.mean() * label_smooth


class Encoder(nn.Module):

    def __init__(self, field, args, causal=False):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [EncoderLayer(args, causal) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)
        self.field = field
        self.d_model = args.d_model
        self.share_embeddings = args.share_embeddings

    def forward(self, x, mask=None):

        encoding = [x]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
            encoding.append(x)
        return encoding


class Decoder(nn.Module):

    def __init__(self, field, args, causal=True,
                positional=False, noisy=False):

        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(args, causal, positional, noisy)
            for i in range(args.n_layers)])

        self.dropout = nn.Dropout(args.drop_ratio)
        self.d_model = args.d_model
        self.field = field
        self.length_ratio = args.length_ratio
        self.positional = positional
        self.cross_attn_fashion = args.cross_attn_fashion

    def forward(self, x, encoding, mask_src=None, mask_trg=None):

        x = self.dropout(x)
        if self.cross_attn_fashion == 'reverse':
            encoding = encoding[1:][::-1]
        elif self.cross_attn_fashion == 'last_layer':
            encoding = [encoding[-1] for _ in len(self.layers)]
        else:
            pass

        for l, (layer, enc) in enumerate(zip(self.layers, encoding[1:])):
            x = layer(x, enc, mask_src=mask_src, mask_trg=mask_trg)
        return x

    def greedy(self, io_dec, encoding, mask_src=None, mask_trg=None):

        encoding = encoding[1:]
        B, T, C = encoding[0].size()  # batch-size, decoding-length, size
        T *= self.length_ratio

        outs = Variable(encoding[0].data.new(B, T + 1).long().fill_(
                    self.field.vocab.stoi['<init>']))
        hiddens = [Variable(encoding[0].data.new(B, T, C).zero_())
                    for l in range(len(self.layers) + 1)]
        # embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])

        eos_yet = encoding[0].data.new(B).byte().zero_()

        for t in range(T):
            
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t] + io_dec.i(outs[:, t], pos=False))

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
        encoding = encoding[1:]
        W = width
        B, T, C = encoding[0].size()

        # expanding
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(
                B, W, T, C).contiguous().view(B * W, T, C)
        mask_src = mask_src[:, None, :].expand(B, W, T).contiguous().view(B * W, T)

        T *= self.length_ratio
        outs = Variable(encoding[0].data.new(B, W, T + 1).long().fill_(
            self.field.vocab.stoi['<pad>']))
        outs[:, :, 0] = self.field.vocab.stoi['<init>']

        logps = Variable(encoding[0].data.new(B, W).float().fill_(0))  # scores
        hiddens = [Variable(encoding[0].data.new(B, W, T, C).zero_())  # decoder states: batch x beamsize x len x h
                    for l in range(len(self.layers) + 1)]
        # embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].data.new(B, W).byte().zero_()  # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(INF)[:, :, None].expand(B, W, W)  # --- big bug, logps < 0
        eos_mask[:, :, 0] = 0  # batch x beam x beam

        for t in range(T):
            hiddens[0][:, :, t] = self.dropout(
                hiddens[0][:, :, t] + io_dec.i(outs[:, :, t], pos=False))
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
            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs))
            outs[:, :, t + 1] = topk_token_inds
            topk_beam_inds = topk_beam_inds[:, :, None, None].expand_as(
                hiddens[0])
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
        
        if args.pry_io:
            self.io_dec = PryIO(trg, args)
            self.io_enc = IO(src, args)
        else:
            self.io_dec = IO(trg, args)
            self.io_enc = IO(src, args)

        if args.share_embeddings:
            self.io_enc.out.weight = self.io_dec.out.weight

        self.fields = {'src': src, 'trg': trg}
        self.args = args

    def prepare_masks(self, inputs):
        field, text = inputs
        if text.ndimension() == 2:  # index inputs
            masks = (text.data != self.fields[field].vocab.stoi['<pad>']).float()
        else:                         # one-hot vector inputs
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

    # def apply_mask(self, inputs, mask, p=1):
    #     outputs = inputs * mask + (1 + (-1) * mask) * p
    #     return outputs

    # def apply_mask_cost(self, loss, mask, return_batch_loss=False):
    #     cost = (loss * mask).sum() / (mask.sum() + TINY)
    #     if not return_batch_loss:
    #         return cost

    #     loss = loss.sum(1, keepdim=True) / (TINY + mask.sum(1, keepdim=True))
    #     return cost, loss

    # def batched_cost(self, decoder_targets, decoder_masks, probs, return_batch_loss=False):
    #     # get loss in a batch-mode

    #     if decoder_targets.ndimension() == 2:  # batch x length
    #         loss = -torch.log(probs + TINY).gather(2, decoder_targets[:, :, None])[:, :, 0]  # batch x length
    #     else:
    #         loss = -(torch.log(probs + TINY) * decoder_targets).sum(-1)
    #     return self.apply_mask_cost(loss, decoder_masks, return_batch_loss)

