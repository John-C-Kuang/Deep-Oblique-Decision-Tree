# global
import pandas as pd

# local
from collections import Counter
import re
from typing import List


class Vectorization:
    def __init__(self, lower: bool = True, top_k: int = -1, internal: Counter = None,
                 *,
                 filter_words: List[str] = None, remove_repetitive: bool = False,
                 filter_punc: bool = True, filter_num: bool = True, filter_len: int = 0):
        """
        Constructor for a stateful Vectorization instance.

        @param lower: boolean flag for converting all words to lower case.
        @param top_k: number of features to be selected with the highest frequencies.
        @param internal: precomputed vocabulary list to be stored.
        @param filter_words: list of words to be filtered out.
        @param remove_repetitive: boolean flag for removing repetitive words in each sentence.
        @param filter_punc: boolean flag for removing punctuations.
        @param filter_num: boolean flag for removing numeric values.
        @param filter_len: words with length less than or equal to be removed.
        """
        self.vocab_list = internal
        self.word_list = None
        if internal is not None:
            if top_k > 0:
                self.word_list = sorted(list(self.vocab_list.most_common(top_k)))
            else:
                self.word_list = sorted(list(self.vocab_list))

        self.lower = lower
        self.k = top_k
        self.remove_repetitive = remove_repetitive
        self.filter_words = set()
        if filter_words is not None:
            self.filter_words = self.filter_words | set(filter_words)
        self.filter_punc = filter_punc
        self.filter_num = filter_num
        self.filter_len = filter_len

    def __repr__(self):
        return self.vocab_list.__repr__()

    def __getitem__(self, item):
        return self.vocab_list.__getitem__(item)

    def __getattr__(self, item):
        if not hasattr(self.vocab_list, item):
            raise AttributeError("Class internal 'Counter' object has no attribute '{}'".format(item))

        return getattr(self.vocab_list, item)

    def __len__(self):
        return self.vocab_list.__len__()

    def fit_on_series(self, series: pd.Series) -> None:
        """
        Fits the vocabulary list and word list for Vectorization on given series.

        @param series: series of sentences to extract vocabularies on.
        @return: None
        """
        if self.remove_repetitive:
            if self.lower:
                series = series.apply(lambda s: s.lower())
            if self.filter_punc:
                series = series.apply(lambda s: re.sub(r'[^\w\s]', ' ', s))
            if self.filter_num:
                series = series.apply(lambda s: re.sub(r'\d', '', s))

            def remove_repetitive(entry: str):
                words = set(entry.split())
                return ' '.join(words)
            series = series.apply(remove_repetitive)

            text = ' '.join(series)
            self.fit_on_texts(text, _class_call=True)

        else:
            text = ' '.join(series)
            self.fit_on_texts(text)

    def fit_on_texts(self, text: str, _class_call: bool = False) -> None:
        """
        Fits the vocabulary list and word list for Vectorization on given sentence.

        @param text: sentence to extract vocabularies on.
        @param _class_call: internal flag indicates if called as a helper function.
        @return: None
        """
        if not _class_call:
            if self.lower:
                text = text.lower()

            if self.filter_punc:
                text = re.sub(r'[^\w\s]', ' ', text)

            if self.filter_num:
                text = re.sub(r'\d', '', text)

        words = [_ for _ in text.split() if _ not in self.filter_words and len(_) > self.filter_len]
        self.vocab_list = Counter(words)
        self.top_k(self.k)

    def top_k(self, k: int) -> None:
        """
        Sets the top k value for the instance and updates the selected word list.

        @param k: number of words selected with the highest frequencies.
        @return: None
        """
        if self.vocab_list is not None:
            if k < 0:
                self.word_list = sorted(list(self.vocab_list))
            else:
                self.word_list = sorted(list(self.vocab_list.most_common(k)))

        self.k = k

    def vectorize(self, text: str, use_frequency: bool = False) -> tuple[int]:
        """
        Vectorize the given sentence with selected word list.

        @param text: sentence to be vectorized.
        @param use_frequency: boolean flag for vectorizing using frequency.
        @return: vectorized feature vector.
        """
        text = text.split()
        if use_frequency:
            count = Counter(text)
            return tuple([count[_] for _ in self.word_list])
        else:
            return tuple([int(_ in text) for _ in self.word_list])
