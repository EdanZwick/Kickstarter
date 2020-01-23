from collections import defaultdict
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

from kickstarter.transformers._base.base_transformer import BaseTransformer

COLUMNS = ["creator_unsuccesses", "creator_successes", "creator_past_proj"]

FAILURE = 0
SUCCESS = 1
TOTAL = 2


class CreatorTransformer(BaseTransformer):
    def __init__(self) -> None:
        self._history = defaultdict(lambda: [0, 0, 0])

    @property
    def input_fields(self) -> List[str]:
        return ["creator_id", "state"]

    def fit(self, x: pd.DataFrame, y: pd.Series):
        creator_history = x['creator_id'].value_counts()
        for index, row in tqdm(x.iterrows()):
            creator_id, state = row["creator_id"], row["state"]
            if creator_history[creator_id] > 1:
                self._history[creator_id][FAILURE] += int(state != "successful")
                self._history[creator_id][SUCCESS] += int(state == "successful")

        for creator_id in tqdm(creator_history.index):
            if self._history[creator_id][FAILURE] + self._history[creator_id][SUCCESS] == 0:
                continue

            if self._history[creator_id][FAILURE] == 0:
                self._history[creator_id][SUCCESS] -= 1
            if self._history[SUCCESS] == 0:
                self._history[creator_id][FAILURE] -= 1
            else:  # If both are above 0 then reduce at random
                if np.random.rand() > 0.5:
                    self._history[creator_id][SUCCESS] -= 1
                else:
                    self._history[creator_id][FAILURE] -= 1

            self._history[creator_id][TOTAL] = self._history[creator_id][FAILURE] + self._history[creator_id][SUCCESS]

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        creator_data = list(x["creator_id"].apply(lambda id: self._history[id]).values)
        return pd.DataFrame(creator_data, index=x.index, columns=COLUMNS)

#
# def get_creator_history(X, y):
#     counts = X['creator_id'].value_counts()
#     # subtract one as we don't want to count the current project
#     get_history = lambda row: counts[row['creator_id']] - 1
#     X['creator_past_proj'] = X.apply(get_history, axis=1)
#     winners = X.loc[X['state'] == 'successful']
#     win_counts = winners['creator_id'].value_counts()
#     X['creator_successes'] = X.apply(_get_success, axis=1, counts=win_counts)
#     # Tenerary clause if due to the fact that if the project is successful, it was excluded twice (in total count and in succ. count)
#     get_unsuc = lambda row: row['creator_past_proj'] - row['creator_successes'] if row['state'] == 'successful' else \
#         row['creator_past_proj'] - row['creator_successes']
#     X['creator_unsuccesses'] = X.apply(get_unsuc, axis=1)
#
#
# def _get_success(row, counts=None):
#     try:
#         # we subtract the 1 as we don't want to consider the current project twards project's history.
#         return counts[row['creator id']] - 1
#     except KeyError:
#         return 0
