""" use for sub-words """
from subprocess import PIPE, Popen

_TOKENIZERS = {}


def register_tokenizers(name):
    def register_fn(fn):
        if name in _TOKENIZERS:
            raise ValueError('Cannot register duplicated models')
        if not callable(fn):
            raise ValueError('Models must be callable ({name})')
        _TOKENIZERS[name] = fn
        return fn

    return register_fn


def get_tokenizer(name):
    if name in _TOKENIZERS:
        return _TOKENIZERS[name]
    else:
        raise ValueError('Cannot find the specific tokenization')


@register_tokenizers('space')
class space_tokenizer(object):
    def __init__(self, space=" ", **kwargs):
        self.space = space

    def tokenize(self, x):
        if not isinstance(x, list):
            return self._tokenize([x])[0]
        else:
            return self._tokenize(x)

    def reverse(self, x):
        if len(x) == 0:
            return ""

        if not isinstance(x[0], list):
            return self._reverse([x])[0]
        else:
            return self._reverse(x)

    def _tokenize(self, x):
        return [u.split(self.space) for u in x]

    def _reverse(self, x):
        return [self.space.join(u) for u in x]


@register_tokenizers('char')
class character_tokenizer(space_tokenizer):
    def _tokenize(self, x):
        return [list(u) for u in x]

    def _reverse(self, x):
        return [''.join(u) for u in x]


@register_tokenizers('byte')
class byte_tokenizer(space_tokenizer):
    def _tokenize(self, x):
        bytes = [u.encode('utf-8').hex() for u in x]
        return [
            byte[k:k + 2] for byte in bytes for k in range(0, len(byte), 2)
        ]

    def _reverse(self, x):
        return [self._smart_decode_hex_str(''.join(u)) for u in x]

    def _decode_hex_str(self, hex_str):
        try:
            return bytes.fromhex(hex_str).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return ''

    def _smart_decode_hex_str(self, hex_str):
        output = self._decode_hex_str(hex_str)

        if output == '':
            # DP the best recovery (max valid chars) if it's broken
            n_bytes = len(hex_str) // 2
            f = [0 for _ in range(n_bytes + 1)]
            pt = [0 for _ in range(n_bytes + 1)]
            for i in range(1, n_bytes + 1):
                f[i], pt[i] = f[i - 1], i - 1
                for j in range(1, min(4, i) + 1):
                    if f[i - j] + 1 > f[i] and len(
                            self._decode_hex_str(hex_str[2 *
                                                         (i - j):2 * i])) > 0:
                        f[i], pt[i] = f[i - j] + 1, i - j
            cur_pt = n_bytes
            while cur_pt > 0:
                if f[cur_pt] == f[pt[cur_pt]] + 1:
                    output = self._decode_hex_str(
                        hex_str[2 * pt[cur_pt]:2 * cur_pt]) + output
                cur_pt = pt[cur_pt]
        return output


@register_tokenizers('bpe')
class bpe_tokenizer(space_tokenizer):
    def __init__(self, codes=None, vocab=None, **kwargs):
        super().__init__()
        self.codes = codes
        self.vocab = vocab

    def _tokenize(self, x):
        if self.vocab is None:
            # use normal white-space detoknizer
            return super()._tokenize(x)
        raise NotImplementedError

    def _reverse(self, x):
        return [' '.join(u).replace('@@ ', '') for u in x]


@register_tokenizers('wordpiece')
class wp_tokenizer(space_tokenizer):
    def __init__(self, vocab=None, **kwargs):
        self.vocab = vocab

    def _tokenize(self, x):
        raise NotImplementedError

    def _reverse(self, x):
        return [u.replace(' ##', '').split() for u in x]


@register_tokenizers('kytea')
class kytea_tokenizer(space_tokenizer):
    def __init__(self, **kwargs):
        try:
            self.tokenizer = Popen(
                'kytea', stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except FileNotFoundError:
            raise (
                "Please install kytea from http://www.phontron.com/kytea/ inorder to evaluate Japanese outputs"
            )

    def _tokenize(self, x):
        return [[
            oi.split('/')[0] for oi in o.split()
        ] for o in self.tokenizer.communicate(("\n".join(x) + "\n").encode(
            "utf-8"))[0].decode('utf-8').split('\n')]
        return outputs

    def _reverse(self, x):
        return ["".join(o) for o in x]
