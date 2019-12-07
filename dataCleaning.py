import os
import pandas as pd
from datetime import datetime
import json
import re


# Our data is originaly split amongst many small csv files.
# This method creats a single data frame from all of them.
def make_dataframe(path=r'rawData', out=None,
                   cache='rick.pickle'):
    """
    Creates a unified panda dataframe from all files inside the folder dir.
    Note that this means that all files in the directory must be valid csv files.
    Default location for csv files is subdirectory on cwd named rawData.
    Default location for pickle cache is rick.pickle file in cwd.
    :param cache: Path to pickle of dataframe or None if no cache is to be used.
    :param pickle: Whether to pickle the output file.
    :param out: (optional) full path for resulting csv location. if none is passed, new csv is not saved.
    :param path: Path to location of csv files.
    :return: panda dataframe made of the unified files.
    """
    if cache is not None and os.path.isfile(cache):
        df = pd.read_pickle(cache)
        print('read dataframe from cache', cache)
        return df
    chunk_list = []
    for fileName in os.listdir(path):
        if fileName.endswith('.csv'):
            chunk = pd.read_csv(os.path.join(path, fileName))
            chunk_list.append(chunk)
    print('Read ', len(chunk_list), 'csv files')
    df = pd.concat(chunk_list, ignore_index=True)
    if out is not None:
        df.to_csv(path_or_buf=out, chunksize=10000)
    if cache is not None:
        df.to_pickle(path=cache)
        print('Saved data frame to', cache)
    return df


def convert_time(df, timefields):
    for col in timefields:
        df[col] = df[col].apply(datetime.utcfromtimestamp)


def convert_goal(df):
    df['goal'] = df.apply(lambda row: round(row['goal']*row['fx_rate']), axis=1)
    df.drop(columns='fx_rate', inplace=True)
    ind = df[df['goal'] == 0].index
    df.drop(ind, inplace=True)

def extract_creator(df):
    pat = '\A{"id":([0-9]*),'
    ids = df['creator']
    ids = ids.map(lambda x: int(re.search(pat, x).group(1)))
    df['creator'] = ids


def extract_catagories(df):
    cats = df['category']
    cats = cats.apply(json.loads)
    mcat = cats.apply(lambda x: int(x.get('parent_id', 0)))
    cats = cats.apply(lambda x: int(x.get('id', 0)))
    df['category'] = cats
    df['parent_category'] = mcat


def remove_duplicates(df):
    df.sort_values(by='state_changed_at', ascending=False, na_position='first', inplace=True)
    df.drop_duplicates(subset='id', inplace=True)


def fix_state(df):
    ind = df[df['state'] == 'live'].index
    df.drop(ind, inplace=True)
    stat = df['state'].map(lambda x: x if x == 'successful' else 'failed')
    df['state'] = stat

#Adds ratio between goal and collected.
def add_ratio(df):
    df['ratio'] = df.apply(lambda row: row['converted_pledged_amount'] / row['goal'], axis=1)


if __name__ == '__main__':
    df = make_dataframe()
    extract_catagories(df)
