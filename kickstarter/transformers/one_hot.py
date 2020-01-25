from typing import List

import pandas as pd

from kickstarter.transformers._base.base_transformer import BaseTransformer


class OneHotTransformer(BaseTransformer):
    @property
    def input_fields(self) -> List[str]:
        return ["country", "category_name", "parent_category_name"]

    def fit(self, x: pd.DataFrame, y: pd.Series):
        pass

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([
            pd.get_dummies(x['country'], prefix='country', dummy_na=True),
            pd.get_dummies(x['category_name'], prefix='category_name', dummy_na=True),
            pd.get_dummies(x['parent_category_name'], prefix='parent_category_name', dummy_na=True)
        ], axis=1)
