from typing import List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from kickstarter.transformers._base.base_transformer import BaseTransformer

LABEL = 'state'


class Data:
    def __init__(self, df: pd.DataFrame, input_fields: List[str] = None, label: str = LABEL) -> None:
        self.le = LabelEncoder()
        self.le.classes_ = np.array(['failed', 'successful', 'suspended', 'canceled'])  # To set FAIL=0 SUCCESS=1
        assert all(self.le.transform(["failed", "successful"]) == [0, 1])
        self.input_fields = input_fields
        self.label = label
        self.train_df, self.test_df = train_test_split(df, test_size=0.2, random_state=42)

    @property
    def df(self):
        return pd.concat([self.train_df, self.test_df])

    @property
    def train_x(self) -> pd.DataFrame:
        if self.input_fields is None:
            return self.train_df.loc[:, self.train_df.columns != self.label]
        else:
            return self.train_df[self.input_fields]

    @property
    def test_x(self) -> pd.DataFrame:
        if self.input_fields is None:
            return self.test_df.loc[:, self.test_df.columns != self.label]
        else:
            return self.test_df[self.input_fields]

    @property
    def train_y(self) -> pd.Series:
        return pd.Series(self.le.transform(self.train_df[self.label]), index=self.train_df.index)

    @property
    def test_y(self) -> pd.Series:
        return pd.Series(self.le.transform(self.test_df[self.label]), index=self.test_df.index)

    def apply_transformer(self, transformer: BaseTransformer) -> None:
        fields = transformer.input_fields
        transformed_train = transformer.fit_transform(self.train_df[fields], self.train_y)

        if isinstance(fields, str):  # Label not sent to transform
            if fields not in self.test_x.columns:
                raise ValueError(f"{fields} not in columns: {self.test_x.columns}")
        else:
            fields = [field for field in fields if field in self.test_x.columns]

        transformed_test = transformer.transform(self.test_x[fields])
        self.train_df = pd.concat([self.train_df, transformed_train], axis=1)
        self.test_df = pd.concat([self.test_df, transformed_test], axis=1)
