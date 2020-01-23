import os
from datetime import datetime
import json
import re
import urllib.request
from urllib.error import HTTPError
import shutil
import zipfile
import pandas as pd
import numpy as np
import scipy.stats
from dataset_generations import datasets
from kickstarter.logger import logger

caches_urls = {
'rick.pickle' : 'https://github.com/EdanZwick/kickstarter-resources/releases/download/1/rick.pickle.zip',
'with_NIMA.pickle' : 'https://github.com/EdanZwick/kickstarter-resources/releases/download/1.1/with_NIMA.pickle.zip'}

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
            download_extract(key)  # Downloads this data set as zip and extracts it.
            logger.debug('Merging {}'.format(key))
            new = makeSingleDf(path + '/' + key)  # Merges the 50+- CSVs in this generation to a single df
            # We used to do this later, but it seems that there are too much lives in this new dataset We now will
            # drop them so if there is any finalized version, it will win de-duping (was already not supposed to happen)
            df = pd.concat([df, new], ignore_index=True, sort=True)
            remove_duplicates(df)
            erase(key)
        except Exception as e:
            with open('bad_data_sets.txt', 'a+') as f:
                f.write(key + '\n')
            erase(key)
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
    cache = os.path.join(cache_folder,cache)
    if not os.path.isdir(cache_folder):
            os.makedirs(cache_folder)
    if not os.path.isfile(cache):
        try:
            download_cache(cache)
        except Exception as e:
            logger.exception(e)
            logger.error('No such pickle file exists on your computer or web')
            return None
    df = pd.read_pickle(cache)
    logger.info('read dataframe from cache', cache, sep=' ')
    return df


def makeSingleDf(path):
    chunk_list = []
    for fileName in sorted(os.listdir(path)):  # Sort to be deterministic
        if fileName.endswith('.csv'):
            chunk = pd.read_csv(os.path.join(path, fileName))
            chunk_list.append(chunk)
    logger.info('Read ', len(chunk_list), 'csv files')
    df = pd.concat(chunk_list, ignore_index=True)
    return df


def downloadData(path=r'rawData/'):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for generation in datasets:
        downloadFile(path + generation + '.zip', datasets[generation])
        logger.info('Downloaded', generation, sep=' ')


def downloadFile(fileName, url, path=r'rawData/'):
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


def download_extract(generation, path=r'rawData/'):
    downloadFile(path + generation + '.zip', datasets[generation])
    logger.info('extracting')
    extract(path + generation)

    
def download_cache(cache):
    name = os.path.split(cache)[-1]
    url = caches_urls.get(name,None)
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
    
    
def erase(generation, path=r'rawData/'):
    fileName = path + generation + '.zip'
    if os.path.isfile(fileName):
        os.remove(fileName)
    if os.path.exists(path + generation + '/'):
        shutil.rmtree(path + generation + '/')
        logger.info('erased {}'.format(generation))


def extract(fileName):
    directory = fileName + '/'
    if os.path.exists(directory):
        logger.info(directory, ' exists', sep=' ')
        return
    os.makedirs(directory)
    with zipfile.ZipFile(fileName + '.zip', 'r') as zip_ref:
        zip_ref.extractall(directory)


def convert_time(df, timefields):
    for col in timefields:
        df[col] = df[col].apply(datetime.utcfromtimestamp)


def extract_month_and_year(df, timefields):
    for col in timefields:
        months = []
        years = []
        for row in df[col]:
            months.append(row.month)
            years.append(row.year)
        df[col + '_month'] = months
        df[col + '_year'] = years


def convert_goal(df):
    df['goal'] = df.apply(lambda row: round(row['goal'] * row['static_usd_rate']), axis=1)
    df.drop(columns='static_usd_rate', inplace=True)
    ind = df[df['goal'] == 0].index
    df.drop(ind, inplace=True)


def extract_creator_id(df):
    # jsons in table are broken, as user names contain quotes or commas which are not escaped in json (kickstarter, Realy?!)
    # Note to self, register to kickstarter as '; * drop tables --'
    #escape_quotes = lambda row: re.sub(r'([^:,{}])(")([^:,{}])',r'\1\"\3',row.creator)
    escape_quotes = lambda row: re.sub(r'("[^"]*":)("[^"\\]*)(")([^"]*)(")([^"\\]*",)',r'\1\2\"\4\"\6',row.creator)
    df['creator'] = df.apply(escape_quotes , axis=1)
    escape_uneven_quotes = lambda row: re.sub(r'([A-Za-z])"([A-Za-z])',r'\1\"\2',row.creator)
    df['creator'] = df.apply(escape_uneven_quotes , axis=1)
    #parse creator id out of the now legal json
    df['creator_id'] = df.apply(get_creator_id , axis=1)


# seperated to enable error handling
def get_creator_id(row):
    try:
        return json.loads(row['creator'])['id']
    except:
        return -1

    
def extract_creator_fields(df):
    get_registration_stat = lambda row: json.loads(row.creator).get('is_registered','unknown')
    df['creator_status'] = df.apply(get_registration_stat , axis=1)
    get_creator_photo = lambda row: json.loads(row.creator).get('avatar').get('medium')
    df['creator_photo'] = df.apply(get_creator_photo , axis=1)
    get_superbacker = lambda row: json.loads(row.creator).get('is_superbacker','unkown')
    df['super_creator'] = df.apply(get_superbacker , axis=1)

    
def get_creator_history(df):
    counts = df['creator id'].value_counts()
    # subtract one as we don't want to count the current project
    get_history = lambda row : counts[row['creator id']] - 1
    df['creator_past_proj'] = df.apply(get_history, axis=1)
    winners = df.loc[df['state'] == 'successful']
    win_counts = winners['creator id'].value_counts()
    df['creator_successes'] = df.apply(get_succes, axis=1, counts = win_counts)
    # Tenerary clause if due to the fact that if the project is successful, it was excluded twice (in total count and in succ. count)
    get_unsuc = lambda row : row['creator_past_proj'] - row['creator_successes'] if row['state'] == 'successful' else row['creator_past_proj'] - row['creator_successes']
    df['creator_unsuccesses'] = df.apply(get_unsuc, axis=1)
    
def get_succes(row, counts = None):
    try:
        # we subtract the 1 as we don't want to consider the current project twards project's history.
        return counts[row['creator id']] - 1
    except KeyError:
        return 0
    
    
def extract_catagories(df):
    cats = df['category']
    cats = cats.apply(json.loads)

    parent_cats_ids = []
    parent_cats_names = []
    cats_ids = []
    cats_names = []

    for cat in cats:
        parent_cats_ids.append(int(cat.get('parent_id', 0)))
        parent_cats_names.append(cat.get('slug', '/').split('/')[0].lower())
        cats_ids.append(int(cat.get('id', 0)))
        cats_names.append(cat.get('name', "").lower())

    df['category'] = cats_ids
    df['parent_category'] = parent_cats_ids
    df['category_name'] = cats_names
    df['parent_category_name'] = parent_cats_names


def remove_duplicates(df):
    df.sort_values(by='state_changed_at', ascending=False, na_position='first', inplace=True)
    df.drop_duplicates(subset='id', inplace=True)


def fix_state(df):
    df.drop(df[~df.state.isin(['successful','failed'])].index, inplace=True)


def extract_country(df):
    # handle cases where there is no location field (just take from country)
    fix_nan = lambda row: '{{"country" : "{}", "state":"unknown"}}'.format(row['country']) if (isinstance(row.location, float) and np.isnan(row.location)) else row['location']
    df['location'] = df.apply(lambda row: fix_nan(row), axis=1)
    # Extract the country column from json
    country_from_json = lambda j : json.loads(j)['country']
    df['country'] = df.apply(lambda row: country_from_json(row['location']), axis=1)


def unify_countries(df, counts, threshold):
    unify_less_frequent = lambda row: 'Global' if counts[row['country']]<=threshold else row['country']
    df['country'] = df.apply(lambda row: unify_less_frequent(row), axis=1)
    
    
def get_us_state(df):
    states = set(['WA', 'DE', 'DC', 'WI', 'WV', 'HI', 'FL', 'FM', 'WY', 'NH', 'NJ', 'NM', 'TX', 'LA', 'NC', 'ND', 'NE', 'TN', 'NY', 'PA', ' V', 'CA', 'NV', 'VA', 'GU', 'CO', 'PW', 'AK', 'AL', 'AS', 'AR', 'VT', 'IL', 'GA', 'IN', 'IA', 'OK', 'AZ', 'ID', 'CT', 'ME', 'MD', 'MA', 'OH', 'UT', 'MO', 'MN', 'MI', 'MH', 'RI', 'KS', 'MT', 'MP', 'MS', 'PR', 'SC', 'KY', 'OR', 'SD'])
    fix_nan = lambda row: '{"state":"unknown"}' if (isinstance(row.location,float) and np.isnan(row.location) and row.country == 'US') else row.location
    df['location'] = df.apply(lambda row: fix_nan(row), axis=1)
    state_from_json = lambda j : json.loads(j)['state'] if json.loads(j)['state'] is not None and json.loads(j)['state'] in states else 'unknown'
    df['country'] = df.apply(lambda row: 'US ' + state_from_json(row['location']) if row['country']=='US' else row['country'], axis=1)

def encode_string_enums(df, col, str_values, number_values):
    mapping = {str_val: number_val for str_val, number_val in zip(str_values, number_values)}
    df = df[col] = df[col].map(mapping)


def add_destination_delta_in_days(df):
    delta = lambda r: (r['deadline'] - r['launched_at']).components.days
    df['destination_delta_in_days'] = df.apply(lambda row: delta(row), axis=1)


def get_image_url(df):
    imgs = df['photo']
    imgs = imgs.apply(json.loads)
    imgs = imgs.map(lambda x: x.get('full', 0))
    df['photo'] = imgs


def download_photos(df, url_column = 'photo', name_column = 'id', folder='tmp'):
    folder = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info('created folder')
    for i, (url, idnum) in enumerate(zip(df[url_column], df[name_column])):
        if (i%10000 == 0):
            logger.info('downloaded {} images'.format(i))
        try:
            with urllib.request.urlopen(url) as response, open(os.path.join(folder, str(idnum)), 'wb+') as out_file:
                shutil.copyfileobj(response, out_file)
        except HTTPError as err: 
            with open('bad_images.txt', 'a+') as f:
                    f.write(str(idnum) + '\n')
                    continue
    logger.info('Downloaded {} images'.format(str(len(df))))
    
    
def erase_photos(folder):
    if folder is None:
        raise ValueError('No path to delete')
    shutil.rmtree(folder)
    

    
def add_nima(df, jsonFile, columnName, image_name_is_project = 'id'):
    if columnName in df.columns:
        logger.info('Data already in dataset!')
        return
    logger.info('opening json')
    with open(jsonFile) as jf:
        scores = json.load(jf)
        logger.info('there are {} recordes in json'.format(len(scores)))
        for i, record in enumerate(scores):
            try:
                iid = int(record.get('image_id'))
                score = record.get('mean_score_prediction')
                df.loc[df[image_name_is_project]==iid, columnName] = score
                if i % 10000 == 0:
                    logger.info ('done with record {}'.format(i))
            except KeyError:
                # if this project was dropped from the dataframe for some reason as the dataset changes.
                logger.info('key error')
                continue
            
                
# Add the pdf value for each score, for each distribution
def add_NIMA_probs(df, succesful_mean, succesful_std, failed_mean, failed_std):
    success_dist = scipy.stats.norm(succesful_mean, succesful_std)
    failed_dist = scipy.stats.norm(failed_mean, failed_std)
    sucess_prob = lambda r: success_dist.pdf(r['nima_score'])
    failed_prob = lambda r: failed_dist.pdf(r['nima_score'])
    df['NIMA_prob_success'] = df.apply(lambda row: sucess_prob(row), axis=1)
    df['NIMA_prob_failed'] = df.apply(lambda row: failed_prob(row), axis=1)
    

def add_NIMA_probs_margin(df):
    margin = lambda r: r['NIMA_prob_success'] - r['NIMA_prob_failed']
    df['NIMA_margin'] = df.apply(lambda row: margin(row), axis=1)
    

def unified_NIMA(df):
    df['NIMA_joined'] = df.apply(lambda row: row['nima_score'] + row['nima_tech'], axis=1)


if __name__ == '__main__':
    df = get_pickles('creators.pickle')
    download_photos(df, url_column = 'creator photo', name_column = 'creator id', folder='creator photos')
    
