"""
Evaluate exact match for Django dataset

Prerequisite:
    `pip install astor==0.6`
"""

try:
    from cStringIO import StringIO
except:
    from io import StringIO

import token as tk
from tokenize import generate_tokens
import ast
import astor


#################################################################################################
#                                                                                               #
#  Wrapper Function                                                                             #
#                                                                                               #
#################################################################################################
from . import register_corpus_metric


@register_corpus_metric('CODE_MATCH')
def get_matching_score(targets, decodes):
    targets = [" ".join(t) for t in targets]
    decodes = [" ".join(d) for d in decodes]
    return computeMatchingAccuracy(decodes, targets)

#################################################################################################


def tokenize_code(code):
    """
    tokenize a given code snippet
    """

    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []

    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break

        tokens.append(tokval)

    return tokens

def reformat_code(code):
    """
    normalize a given code snippet by parsing it to AST and 
    regenerate the surface code from it
    """
    py_ast = ast.parse(code).body[0]
    reformatted_code = astor.to_source(py_ast).strip()

    return reformatted_code


def is_match(hyp_code, ref_code):
    """
    give a hypothesis Python code decoded from the model
    and a gold-standard reference code, compute if it's 
    an exact match

    We first reformat the code, e.g., unifying quotes and indents.

    Args:
        hyp_code: string of hypothesis code
        ref_code: string of gold reference code
        Note: those code should be valid python source code!

    Returns:
        Boolean value indicating if there is a successful match
    """

    norm_hyp_code = reformat_code(hyp_code)
    norm_ref_code = reformat_code(ref_code)

    hyp_code_tokens = tokenize_code(norm_hyp_code)
    ref_code_tokens = tokenize_code(norm_ref_code)

    return ref_code_tokens == hyp_code_tokens


def computeMatchingAccuracy(outputs, targets):
    acc = []
    for o, t in zip(outputs, targets):
        try:
            c = is_match(o, t)
        except Exception:
            c = 0
        acc.append(c)
    return np.mean(acc)