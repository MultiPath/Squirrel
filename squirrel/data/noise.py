# class Symbols(data.Field):
#     def __init__(self,
#                  reverse_tokenize,
#                  shuffle=0,
#                  dropout=0,
#                  replace=0,
#                  additional_tokens=None,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.reverse_tokenizer = reverse_tokenize
#         self.shuffle, self.dropout, self.replace = shuffle, dropout, replace
#         self.additional_tokens = additional_tokens if additional_tokens is not None else []
#         self.name = 'symbols'

#     def build_vocab(self, *args, **kwargs):
#         """Construct the Vocab object for this field from one or more datasets.
#         Arguments:
#             Positional arguments: Dataset objects or other iterable data
#                 sources from which to construct the Vocab object that
#                 represents the set of possible values for this field. If
#                 a Dataset object is provided, all columns corresponding
#                 to this field are used; individual columns can also be
#                 provided directly.
#             Remaining keyword arguments: Passed to the constructor of Vocab.
#         """
#         counter = Counter()
#         sources = []
#         for arg in args:
#             if isinstance(arg, Dataset):
#                 sources += [
#                     getattr(arg, name) for name, field in arg.fields.items()
#                     if field is self
#                 ]
#             else:
#                 sources.append(arg)
#         for data in sources:
#             for x in data:
#                 if not self.sequential:
#                     x = [x]
#                 try:
#                     counter.update(x)
#                 except TypeError:
#                     counter.update(chain.from_iterable(x))
#         specials = list(
#             OrderedDict.fromkeys(tok for tok in [
#                 self.unk_token, self.pad_token, self.init_token, self.eos_token
#             ] + self.additional_tokens if tok is not None))
#         self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

#     def word_shuffle(self, x):
#         if self.shuffle == 0:
#             return x
#         return [
#             x[i] for i in (np.random.uniform(0, self.shuffle, size=(len(x))) +
#                            np.arange(len(x))).argsort()
#         ]

#     def word_dropout(self, x):
#         if self.dropout == 0:
#             return x
#         return [
#             xi for xi, di in zip(x, (
#                 np.random.rand(len(x)) >= self.dropout).tolist()) if di == 1
#         ]

#     def word_blank(self, x, tok='<unk>'):
#         if self.replace == 0:
#             return x
#         return [
#             xi if di == 1 else tok for xi, di in zip(x, (
#                 np.random.rand(len(x)) >= self.replace).tolist())
#         ]

#     def add_noise(self, x, noise_level=None):
#         if noise_level is None:
#             return x

#         if noise_level == 'n1':
#             c = np.random.choice(3)
#         elif noise_level == 'n2':
#             c = np.random.choice(4)
#         elif noise_level == 'n3':
#             c = 4
#         else:
#             raise NotImplementedError

#         if c == 0:
#             return self.word_shuffle(x)
#         elif c == 1:
#             return self.word_dropout(x)
#         elif c == 2:
#             return self.word_blank(x, self.unk_token)
#         elif c == 3:
#             return x
#         elif c == 4:
#             return self.word_blank(
#                 self.word_dropout(self.word_shuffle(x)), self.unk_token)
#         else:
#             raise NotImplementedError

#     def process(self, batch, device=None):
#         padded = self.pad(batch)
#         tensor = self.numericalize(padded, device=device)
#         return tensor

#     def extend_padding(self, batch, maxlen):
#         new_batch = batch.new_zeros(batch.size(0), maxlen).fill_(
#             self.vocab.stoi[self.pad_token])
#         new_batch[:, :batch.size(1)] = batch
#         return new_batch

#     def reverse(self,
#                 batch,
#                 width=1,
#                 return_saved_time=False,
#                 reverse_token=True):
#         if not self.batch_first:
#             batch.t_()

#         with torch.cuda.device_of(batch):
#             batch = batch.tolist()

#         batch = [[self.vocab.itos[ind] for ind in ex]
#                  for ex in batch]  # denumericalize

#         def trim(s, t):
#             sentence = []
#             for w in s:
#                 if w == t:
#                     break
#                 sentence.append(w)
#             return sentence

#         batch = [trim(ex, self.eos_token)
#                  for ex in batch]  # trim past frst eos

#         def filter_special(tok):
#             return tok not in (self.init_token, self.pad_token)

#         def count(ex):
#             n_step = 0
#             n_pad = 0
#             n_word = 0

#             filtered = []
#             decision = []

#             for e in ex:
#                 if e == self.init_token:
#                     continue

#                 if e == self.pad_token:
#                     n_pad += 1
#                     if n_word > 0:
#                         n_step += 1
#                         n_word = 0

#                 else:
#                     if n_word < (width - 1):
#                         n_word += 1

#                     else:
#                         n_word = 0
#                         n_step += 1

#                     if n_word == 1:
#                         decision.append(0)
#                     else:
#                         decision.append(1)

#                     filtered.append(e)

#             saved_time = (n_step + (n_word == 0)) / (1 + len(filtered))
#             accuracy = len(filtered) / (len(ex) + 1e-9)
#             return filtered, saved_time, accuracy, decision

#         if return_saved_time:
#             batch_filtered, saved_time, accuracy, decisions = [], [], [], []
#             for ex in batch:
#                 b, s, a, d = count(ex)
#                 batch_filtered.append(b)
#                 saved_time.append(s)
#                 accuracy.append(a)
#                 decisions.append(d)

#         else:
#             batch_filtered = [list(filter(filter_special, ex)) for ex in batch]

#         if not reverse_token:
#             return batch_filtered

#         output = [self.reverse_tokenizer(ex) for ex in batch_filtered]
#         if return_saved_time:
#             return output, saved_time, accuracy, decisions

#         return output

#     def reapply_noise(self, data, noise):
#         batch = self.reverse(data, reverse_token=False)
#         batch = [self.add_noise(ex, noise) for ex in batch]
#         return self.process(batch, device=data.get_device())
