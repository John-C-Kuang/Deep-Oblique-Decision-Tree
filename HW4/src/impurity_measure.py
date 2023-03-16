from math import log2

def gini(self, cnt: dict):
    """ 1 - sum(p^2) """
    return 1 - sum([(v / self.__total(cnt)) ** 2 for v in cnt.values()])

def entropy(self, cnt: dict):
    tot = self.__total(cnt)
    return -sum([(v / tot) * log2(v / tot) for v in cnt.values()])