import os
import shutil
import urllib.request
import zipfile

import pandas as pd
import urllib.request
from kickstarter.dataset_generations import datasets
from kickstarter.logger import logger

_caches_urls = {
    'rick.pickle': 'https://github.com/EdanZwick/kickstarter-resources/releases/download/1/rick.pickle.zip',
    'with_NIMA.pickle': 'https://github.com/EdanZwick/kickstarter-resources/releases/download/1.2/with_NIMA.pickle.zip'}


# Our data is originally split amongst many small csv files.
# This method creates a single data frame from all of them.
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
    # first try to save time by locating a cache, either pickle or CSV
    if cache is not None:
        df = get_pickles(cache)
        return df
    if out is not None and os.path.isfile(out):
        logger.info('read dataframe from csv file', os.path.join(os.getcwd(), out), sep=' ')
        return pd.read_csv(os.path.join(os.getcwd(), out))
    # log datasets that did not load well, without ending the whole process.
    with open('bad_data_sets.txt', 'w+') as f:
        f.write('bad data sets: \n')
    df = pd.DataFrame()  # Initial dataframe that will be grown
    logger.info('Downloading datasets, expect this to take a few minutes')
    for key in datasets:
        try:
            logger.debug('Downloading {}'.format(key))
            _download_extract(key)  # Downloads this data set as zip and extracts it.
            logger.debug('Merging {}'.format(key))
            new = _make_single_df(path + '/' + key)  # Merges the 50+- CSVs in this generation to a single df
            # We used to do this later, but it seems that there are too much lives in this new dataset We now will
            # drop them so if there is any finalized version, it will win de-duping (was already not supposed to happen)
            df = pd.concat([df, new], ignore_index=True, sort=True)
            _remove_duplicates(df)
            _erase(key)
        except Exception as e:
            with open('bad_data_sets.txt', 'a+') as f:
                f.write(key + '\n')
            _erase(key)
            logger.exception(e)
            continue
    # if any record has null id, erase it:
    df.dropna(subset=['id'], inplace=True)
    logger.info('there are ', len(df.index), ' records in data set', sep='')
    # save united dataset to csv or pickle.
    if out is not None:
        df.to_csv(path_or_buf=out, chunksize=10000)
    if cache is not None:
        df.to_pickle(path=cache)
        logger.info('Saved data frame to', cache)
    return df


def get_pickles(cache):
    cache_folder = 'pickled_data'
    cache = os.path.join(cache_folder, cache)
    if not os.path.isdir(cache_folder):
        os.makedirs(cache_folder)
    if not os.path.isfile(cache):
        try:
            _download_cache(cache)
        except Exception as e:
            logger.exception(e)
            logger.error('No such pickle file exists on your computer or web')
            return None
    df = pd.read_pickle(cache)
    logger.info('read dataframe from cache', cache, sep=' ')
    return df


def _make_single_df(path):
    chunk_list = []
    for fileName in sorted(os.listdir(path)):  # Sort to be deterministic
        if fileName.endswith('.csv'):
            chunk = pd.read_csv(os.path.join(path, fileName))
            chunk_list.append(chunk)
    logger.info('Read ', len(chunk_list), 'csv files')
    df = pd.concat(chunk_list, ignore_index=True)
    return df


def _download_file(fileName, url, path=r'rawData/'):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # fileName = path + fileName
    if os.path.isfile(fileName) or os.path.exists(fileName[:-4]):
        logger.info('file {} already downloaded'.format(fileName))
        return
    with urllib.request.urlopen(url) as response, open(fileName, 'wb+') as out_file:
        shutil.copyfileobj(response, out_file)
    logger.info('downloaded {}'.format(fileName))


def _download_extract(generation, path=r'rawData/'):
    _download_file(path + generation + '.zip', datasets[generation])
    logger.info('extracting')
    _extract(path + generation)


def _download_cache(cache):
    name = os.path.split(cache)[-1]
    url = _caches_urls.get(name, None)
    if url is None:
        raise Exception('No url for file')
    if url.endswith('.zip'):
        # save temp as zip file before extracting
        name += '.zip'
        folder = os.path.split(cache)[0] + '/'
        cache = os.path.join(folder, name)
    # download actual file (zip or pickle)
    with urllib.request.urlopen(url) as response, open(cache, 'wb+') as out_file:
        shutil.copyfileobj(response, out_file)
    # extract pickle from zip and erase zip
    if name.endswith('.zip'):
        with zipfile.ZipFile(cache, 'r') as zip_ref:
            zip_ref.extractall(folder)
        if os.path.isfile(cache):
            os.remove(cache)
    logger.info('downloaded pickle from rescorces')


def _erase(generation, path=r'rawData/'):
    fileName = path + generation + '.zip'
    if os.path.isfile(fileName):
        os.remove(fileName)
    if os.path.exists(path + generation + '/'):
        shutil.rmtree(path + generation + '/')
        logger.info('erased {}'.format(generation))


def _extract(fileName):
    directory = fileName + '/'
    if os.path.exists(directory):
        logger.info(directory, ' exists', sep=' ')
        return
    os.makedirs(directory)
    with zipfile.ZipFile(fileName + '.zip', 'r') as zip_ref:
        zip_ref.extractall(directory)


def _remove_duplicates(df):
    df.sort_values(by='state_changed_at', ascending=False, na_position='first', inplace=True)
    df.drop_duplicates(subset='id', inplace=True)


def downloadData(path=r'rawData/'):  # TODO: unused. delete this
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for generation in datasets:
        _download_file(path + generation + '.zip', datasets[generation])
        logger.info('Downloaded', generation, sep=' ')
