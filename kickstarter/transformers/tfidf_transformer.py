import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from kickstarter.transformers._base.base_transformer import BaseTransformer

STOP_WORDS = set(stopwords.words('english'))


class TfidfTransformer(BaseTransformer):

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(min_df=0.01, stop_words=STOP_WORDS)

    @property
    def input_fields(self) -> str:
        return "blurb"

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self._vec.fit(x.astype("str"))

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        arr = self._vec.transform(x.astype("str")).toarray()
        return pd.DataFrame(arr, index=x.index).add_prefix("tfidf_")
