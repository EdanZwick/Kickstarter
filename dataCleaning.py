import os
import pandas as pd
from datetime import datetime

irrelavent = ['disable_communication', 'friends', 'is_backing']
redundant = ['country_displayable_name', 'currency_symbol', 'currency_trailing_code', 'current_currency']
extras = []


# why fields were let go
#  'country_displayable_name' - redundant
#   'currency_symbol' - redundant
#   'currency_trailing_code' - r.
#   'current_currency' - r.
#   'disable_communication' - very few projects, probably turned off after project failed.
#  'is_backing' - very few records contain info.
#

# explanation for remaining fields:
#

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

if __name__ == '__main__':
    df = make_dataframe()
    timefields = ['created_at', 'deadline', 'launched_at']
    convert_time(df, timefields)
    print(df.loc[1,'created_at'])





    # out=r'/home/ez/PycharmProjects/kickstarter/united.csv'
    # cols = list(df.columns.values)
    # print(cols)
    # y = len(df.index)
    # print('number of records:', y)
    # num_nulls(df)
    # x = df['id'].nunique()
    # print('number of unique values', x, 'meaning there are ', y-x,'duplicates')
