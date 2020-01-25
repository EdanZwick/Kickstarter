import json
import numpy as np
from typing import List

import pandas as pd
from tqdm import tqdm

from kickstarter.logger import logger
from kickstarter.transformers._base.base_transformer import BaseTransformer

JSONS = {
    "nima_score": "NIMA predictions/predictions_imgs_all.json",
    "nima_tech": "NIMA predictions/predictions_imgs_all_technical.json"
}


class NimaTransformer(BaseTransformer):

    def __init__(self) -> None:
        self._nima_scores = {key: _load_scores_dict(JSONS[key]) for key in JSONS}

    @property
    def input_fields(self) -> str or List[str]:
        return "id"

    def fit(self, x: pd.DataFrame, y: pd.Series):
        pass

    def transform(self, x: pd.Series) -> pd.DataFrame:
        result = {}
        for col_name, scores_dict in self._nima_scores.items():
            logger.info('opening json')

            logger.info('there are {} recordes in json'.format(len(scores_dict)))
            nima_records = []
            for iid in tqdm(x):
                try:
                    nima_records.append(scores_dict[iid])
                except KeyError:
                    # if this project was dropped from the dataframe for some reason as the dataset changes.
                    logger.info(f'key error {id}')
                    nima_records.append(np.nan)
            result[col_name] = nima_records
        return pd.DataFrame(result, index=x.index)


def add_nima(df, jsonFile, columnName, image_name_is_project='id'):
    if columnName in df.columns:
        logger.info('Data already in dataset!')
        return
    logger.info('opening json')
    scores_dict = _load_scores_dict(jsonFile)

    logger.info('there are {} recordes in json'.format(len(scores_dict)))
    nima_records = []
    for index, record in tqdm(df.iterrows()):
        try:
            iid = record[image_name_is_project]
            nima_records.append(scores_dict[iid])
        except KeyError:
            # if this project was dropped from the dataframe for some reason as the dataset changes.
            logger.info(f'key error {record[image_name_is_project]}')


def _load_scores_dict(json_file: str) -> dict:
    with open(json_file) as jf:
        scores = json.load(jf)
    scores_dict = {}
    for score in scores:
        scores_dict[int(score["image_id"])] = score["mean_score_prediction"]
    return scores_dict
