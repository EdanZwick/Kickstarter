import os
import pickle
from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import dataCleaning as dc

DF_CACHE = "clean_df.pickle"

LABEL = 'state'
INPUT_FIELDS = ['launched_at_month', 'launched_at_year', 'category', 'parent_category', 'destination_delta_in_days']


class Data:
    def __init__(self, train_x, test_x, train_y, test_y) -> None:
        self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y


@lru_cache()
def get_data() -> Data:
    df = _get_cleaned_df()
    train_x, test_x, train_y, test_y = _split_dataframe(df)

    return Data(train_x, test_x, train_y, test_y)


def _get_cleaned_df() -> pd.DataFrame:
    if os.path.exists(DF_CACHE):
        with open(DF_CACHE, "rb") as f:
            return pickle.load(f)

    df = dc.make_dataframe()
    redundant = ['country_displayable_name', 'currency_symbol', 'currency_trailing_code', 'current_currency',
                 'source_url', 'disable_communication', 'profile', 'urls', 'photo', 'usd_pledged',
                 'usd_type']
    df.drop(columns=redundant, inplace=True)
    empty = ['friends', 'is_backing', 'is_starred', 'permissions']
    df.drop(columns=empty, inplace=True)
    timefields = ['created_at', 'deadline', 'launched_at', 'state_changed_at']
    dc.convert_time(df, timefields)
    dc.extract_creator(df)  # replaces the creator json with creator id int
    dc.extract_catagories(df)  # gets project catagory data
    dc.remove_duplicates(df)
    dc.convert_goal(df)
    dc.extract_month_and_year(df, timefields)
    dc.add_destination_delta_in_days(df)
    # preparing to train
    df = df.loc[df['state'].isin(['failed', 'successful'])]
    dc.encode_string_enums(df, 'state', ['failed', 'successful'], [0, 1])
    dc.set_text_statistics(df)
    dc.set_semantics(df)

    with open(DF_CACHE, "wb") as f:
        pickle.dump(df, f)
    return df


def _split_dataframe(df):
    le = LabelEncoder()
    X = df[INPUT_FIELDS]
    y = le.fit_transform(df[LABEL])

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    return train_x, test_x, train_y, test_y
