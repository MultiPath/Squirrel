"""
python version edit-distance
"""
import numpy as np


def suggested_ed2_path(xs, ys, terminal_symbol=1):
    return [
        edit_distance2_backtracking(x, y, terminal_symbol)
        for (x, y) in zip(xs, ys)
    ]


def compute_ed2(xs, ys):
    return [edit_distance2_with_dp(x, y)[-1, -1] for (x, y) in zip(xs, ys)]


def edit_distance2_with_dp(x, y):
    D = np.arange(len(x) + 1)[:, None] + np.arange(len(y) + 1)[None, :]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                          D[i - 1, j - 1] + 2 * (x[i - 1] != y[j - 1]))
    return D


def edit_distance2_backtracking(x, y, terminal_symbol=1):

    if len(x) == 0:
        return [y, []]

    # compute the distance
    D = edit_distance2_with_dp(x, y)
    i = len(x)
    j = len(y)

    # back-tracking the edits
    seq = []
    while (i >= 0 and j >= 0):
        if (i == 0) and (j == 0):
            break

        if (j > 0) and (D[i, j - 1] < D[i, j]):
            seq.append('INSERT {}'.format(y[j - 1]))
            j = j - 1
        elif (i > 0) and (D[i - 1, j] < D[i, j]):
            seq.append('DELETE {}'.format(x[i - 1]))
            i = i - 1
        else:
            seq.append('KEEP {}'.format(x[i - 1]))
            i = i - 1
            j = j - 1

    def prepare_label(seq, source):
        insert_labels = [[] for _ in range(len(source) + 1)]
        delete_labels = [0 for _ in range(len(source))]

        prev_op = 'NONE'
        s, t = -1, 0
        for i in range(len(seq)):
            op, word = seq[i].split()
            word = int(word)

            if prev_op != 'INSERT':
                s += 1

            if op == 'INSERT':
                insert_labels[s].append(word)
            else:
                if op == 'KEEP':
                    delete_labels[t] = 0
                    t += 1
                elif op == 'DELETE':
                    delete_labels[t] = 1
                    t += 1
                # if prev_op != 'INSERT':
                #     insert_labels[s].append(terminal_symbol)

            prev_op = op

        for s in range(len(insert_labels)):
            if len(insert_labels[s]) == 0:
                insert_labels[s].append(terminal_symbol)
        insert_labels.append(delete_labels)
        return insert_labels

    return prepare_label(seq[::-1], x)
