from math import log2
from collections import Counter

def gini(cnt: Counter):
    """ 1 - sum(p^2) """
    return 1 - sum([(v / sum(cnt.values())) ** 2 for v in cnt.values()])


def entropy(cnt: Counter):
    tot = sum(cnt.values())
    return -sum([(v / tot) * log2(v / tot) for v in cnt.values()])
