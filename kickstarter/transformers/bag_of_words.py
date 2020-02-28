import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from kickstarter.transformers._base.base_transformer import BaseTransformer

BATCH_SIZE = 10000

STOP_WORDS = set(stopwords.words('english'))


class BagOfWords(BaseTransformer):
    def __init__(self) -> None:
        self._vectorizer = CountVectorizer(min_df=0.0001, stop_words=STOP_WORDS)
        self._true_proba = None  # P(y=1|x)
        self._false_proba = None  # P(y=0|x)

    @property
    def input_fields(self) -> str:
        return "blurb"

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._vectorizer.fit(x)
        x_true = x[y == 1]
        x_false = x[y == 0]

        true_counts = np.sum(self._vectorizer.transform(x_true), axis=0) + 1
        false_count = np.sum(self._vectorizer.transform(x_false), axis=0) + 1

        self._true_proba = true_counts / (true_counts + false_count)  # #{y=1,x} / #{x}
        self._false_proba = false_count / (true_counts + false_count)  # #{y=0,x} / #{x}

    def transform(self, x: pd.Series) -> pd.DataFrame:
        result = []
        for i in range(0, len(x) + BATCH_SIZE - 1, BATCH_SIZE):
            transformed_x = self._vectorizer.transform(x[i:i + BATCH_SIZE]).todense()

            true_prob = self._get_proba(transformed_x, self._true_proba)
            false_prob = self._get_proba(transformed_x, self._false_proba)

            result.extend(np.array(true_prob > false_prob, dtype="int"))

        return pd.DataFrame(result, index=x.index, columns=["bag_of_words"])

    @staticmethod
    def _get_proba(transformed_x: np.ndarray, proba_vector: np.ndarray) -> np.ndarray:
        # P(y| #{x_i} ) = P(y | x_i)^#{x_i}
        prob = np.power(proba_vector, transformed_x)

        # P(y | x_1, ..., x_n) = multiply for i ( P(y | x_i)^#{x_i} )
        prob = np.prod(prob, axis=1)

        return np.reshape(np.where(prob == 1, 0, prob), len(prob))
