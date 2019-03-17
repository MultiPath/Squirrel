import numpy as np

COLORS = ['red', 'green', 'yellow', 'blue', 'white', 'magenta', 'cyan']

INF = 1e10
TINY = 1e-9
BIG = 10000000


class NegativeDistanceScore(object):
    def __init__(self):

        # pre-compute some values
        self.scores = {}

        self.scores[0.5] = self.compute_score_full(50, 0.5)
        self.scores[1.0] = self.compute_score_full(50, 1.0)
        self.scores[2.0] = self.compute_score_full(50, 2.0)

    def __call__(self, i, L, tau):
        if tau is None:
            return 1 / L
        if tau in self.scores:
            if L < self.scores[tau].shape[0]:
                return self.scores[tau][L - 1, i]
        return self.compute_score(L, tau)[i]

    def compute_score(self, L, tau):
        s = np.array([-abs(L / 2 - i) / tau for i in range(L)])
        s = np.exp(s - s.max())
        return s / s.sum()

    def compute_score_full(self, L, tau):
        s = -abs(np.arange(0, L - 1)[:, None] / 2 -
                 np.arange(L)[None, :]) / tau
        s = np.tril(s, 0) + np.triu(s - INF, 1)
        s = np.exp(s - s.max(1, keepdims=True))
        return s / s.sum(1, keepdims=True)
