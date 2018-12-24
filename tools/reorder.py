import os
import sys
import numpy as np
import shutil
import pickle
import spacy

from collections import deque

spacy.prefer_gpu()

# helper functions for depdendency parsing

def recover_sents(data):
    outputs = []
    o = []
    for i, d in enumerate(data):
        o.append(i)
        if '@@' not in d:
            outputs.append(o)
            o = []
    
    lines = [''.join([data[oi].replace('@@', '') for oi in o]) for o in outputs]
    return lines, outputs

def dep2path(lines, outs):

    lines = list(lines.sents)
    reordered = []
    caches = deque()
    for l in lines:
        caches.append(l.root)
        while len(caches) > 0:
            a = caches.popleft()
            reordered.append(a.i)
            for c in list(a.children):
                caches.append(c)

    reordered = [j for i in [outs[r] for r in reordered] for j in i]
    return reordered    


CUTOFF = 85 # RO-EN dataset, for other dataset, this value may be different

def reorder(filename, order='l2r'):

    shutil.copy(filename + '.src', filename + '.{}.src'.format(order))

    fi = open(filename + '.trg')
    fo = open(filename + '.{}.trg'.format(order), 'w')
    fp = open(filename + '.{}.pos'.format(order), 'w')

    if ('common' in order ) or ('rare' in order):
        vocab_index, vocab_freq = pickle.load(open(filename + '.trg.voc.pkl', 'rb'))
        vocab_freq = {w[0]: w[1] for w in vocab_freq}

    if order == 'dep':  # get the path following the default path of the depdendency tree (ROOT-LEAF, LEFT-RIGHT)
        nlp = spacy.load('en')
        fd = open(filename + '.{}.full'.format(order), 'w')


    for i, line in enumerate(fi):

        if i % 1000 == 0:
            print('processed {} lines.'.format(i))

        words = line.strip().split()
        positions = list(range(1, len(words) + 1))
        eos_pos = len(words) + 1

        if order == 'l2r':
            words = ['<stop>'] + words
            positions = [0, 5000] + positions + [eos_pos]

        elif order == 'r2l':
            words = ['<stop>'] + words[::-1]
            positions = [0, 5000] + positions[::-1] + [eos_pos]

        elif order == 'odd':
            words = ['<stop>'] + words[::2] + words[1::2]
            positions = [0, 5000] + positions[::2] + positions[1::2] + [eos_pos]

        elif order == 'common':
            positions = [0, 5000] + [positions[i] for i, w in enumerate(words) if vocab_index[w] <= CUTOFF] + [positions[i] for i, w in enumerate(words) if vocab_index[w] > CUTOFF] + [eos_pos]
            words = ['<stop>'] + [w for w in words if vocab_index[w] <= CUTOFF] + [w for w in words if vocab_index[w] > CUTOFF]
        
        elif order == 'rare':
            positions = [0, 5000] + [positions[i] for i, w in enumerate(words) if vocab_index[w] > CUTOFF] + [positions[i] for i, w in enumerate(words) if vocab_index[w] <= CUTOFF] + [eos_pos]
            words = ['<stop>'] + [w for w in words if vocab_index[w] > CUTOFF] + [w for w in words if vocab_index[w] <= CUTOFF]

        elif order == 'common_freq':
            word_order = np.argsort([vocab_freq[w] for w in words])[::-1].tolist()
            words = ['<stop>'] + [words[t] for t in word_order]
            positions = [0, 5000] + [positions[t] for t in word_order] + [eos_pos]
        
        elif order == 'rare_freq':
            word_order = np.argsort([vocab_freq[w] for w in words]).tolist()
            words = ['<stop>'] + [words[t] for t in word_order]
            positions = [0, 5000] + [positions[t] for t in word_order] + [eos_pos]

        elif order == 'dep':

            lines, outs = recover_sents(words)
            lines = nlp.tokenizer.tokens_from_list(lines)  # do not use its internal tokenizer
            nlp.tagger(lines)
            nlp.parser(lines)
            # lines = nlp(lines)
            
            fd.write(' '.join(['{}/{}/{}'.format(d, d.head, d.dep_) for d in lines]) + '\n')
            
            reordered = dep2path(lines, outs)
            words = ['<stop>'] + [words[t] for t in reordered]
            positions = [0, 5000] + [positions[t] for t in reordered] + [eos_pos]

        else:
            raise NotImplementedError
            

        fo.write(' '.join(words) + '\n')
        fp.write(' '.join([str(p) for p in positions]) + '\n')

    print('done.')

reorder(sys.argv[1], sys.argv[2])
