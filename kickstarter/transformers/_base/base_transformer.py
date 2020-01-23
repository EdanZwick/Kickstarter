import abc
from typing import List

import pandas as pd


class BaseTransformer(abc.ABC):

    @property
    @abc.abstractmethod
    def input_fields(self) -> List[str]:
        pass

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series):
        pass

    @abc.abstractmethod
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(x, y)
        return self.transform(x)
