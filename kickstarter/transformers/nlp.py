from typing import List

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from kickstarter.transformers._base.base_transformer import BaseTransformer

COLUMNS = ["blurb_pos", "blurb_neg", "blurb_compound"]


class SemanticTransformer(BaseTransformer):

    def __init__(self) -> None:
        self._analyser = SentimentIntensityAnalyzer()

    @property
    def input_fields(self) -> List[str]:
        return ["blurb"]

    def fit(self, x: pd.DataFrame, y: pd.Series):
        pass

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        neg = []
        pos = []
        compound = []

        for row in x['blurb'].astype(str):
            score = self._analyser.polarity_scores(str(row.encode("utf-8")))
            neg.append(score['neg'])
            pos.append(score['pos'])
            compound.append(score['compound'])

        return pd.DataFrame(zip(pos, neg, compound), index=x.index,
                            columns=COLUMNS)
