# global
import numpy as np

# local
from collections import Counter
from typing import List
import re


class NaiveBayes:
    def __init__(self, cls_counters: List[Counter], cls_occur: List[int], cls_sum: List[int] = None):
        """
        Constructor for a stateful Naive Bayes instance.

        @param cls_counters: list of Counter of class features.
        @param cls_occur: list of int of class occurrences.
        @param cls_sum: list of int for number of total class features.
        """
        if cls_sum is None:
            cls_sum = [sum(_.values()) for _ in cls_counters]
        else:
            if not (len(cls_counters) == len(cls_occur) and len(cls_occur) == len(cls_sum)):
                raise ValueError('Length of class attribute lists must be the same')

        self.cls_attrs = tuple(zip(cls_counters, cls_sum))
        self.cls_occur = cls_occur
        self.cls_probs = np.array([_ / sum(cls_occur) for _ in cls_occur])

    def log_score(self, text: str,
                  *,
                  lower=True, filter_len: int = 0, filter_words: List[str] = None,
                  remove_repetitive: bool = False, remove_punc: bool = True, remove_num: bool = True,
                  safety_factor: float = 1e-7) -> np.ndarray:
        """
        Calculates the Naive Bayes Log Scores on the given sentence.

        @param text: sentence to be predicted.
        @param lower: boolean flag for converting all words to lower case.
        @param filter_len: words with length less than or equal to be removed.
        @param filter_words: list of words to be filtered out.
        @param remove_repetitive: boolean flag for removing repetitive words in each sentence.
        @param remove_punc: boolean flag for removing punctuations.
        @param remove_num: boolean flag for removing numeric values.
        @param safety_factor: minimal value to prevent divided by 0 warning.
        @return: array of calculated Naive Bayes Log Scores corresponding to each class.
        """
        # string preprocess
        if lower:
            text = text.lower()
        if remove_punc:
            text = re.sub(r'[^\w\s]', ' ', text)
        if remove_num:
            text = re.sub(r'\d', '', text)

        if remove_repetitive:
            words = set(text.split())
            text = ' '.join(words)

        filter_list = set()
        if filter_words is not None:
            filter_list = filter_list | set(filter_words)

        probs = np.zeros_like(self.cls_occur, dtype=np.float32) + safety_factor

        for _ in text.split():
            if _ not in filter_list and len(_) > filter_len:
                cond_prob = np.array([float(cnt[0][_] + 1) / cnt[1] for cnt in self.cls_attrs])
                probs += np.log(cond_prob)

        log_cls_probs = np.log(self.cls_probs)
        return probs + log_cls_probs
