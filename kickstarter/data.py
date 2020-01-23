from typing import List

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LABEL = 'state'


class Data:
    def __init__(self, df: pd.DataFrame, input_fields: List[str] = None, label: str = LABEL) -> None:
        self.le = LabelEncoder()
        y = self.le.fit_transform(df[label])

        if input_fields is None:
            self.input_fields = list(df.columns.values)
            self.input_fields.remove(label)

        self._train_x, self._test_x, self.train_y, self.test_y = train_test_split(df, y, test_size=0.2)

    @property
    def train_x(self) -> pd.DataFrame:
        return self._train_x[self.input_fields]

    @train_x.setter
    def train_x(self, value: pd.DataFrame):
        self._train_x = value

    @property
    def test_x(self) -> pd.DataFrame:
        return self._test_x[self.input_fields]

    @test_x.setter
    def test_x(self, value: pd.DataFrame):
        self._test_x = value

    def apply(self, callable):
        callable(self.train_x)
        callable(self.test_x)
