import os
from datetime import datetime
import json
import re
import urllib.request
import shutil
import zipfile
import numpy as np
import pandas as pd
from dataset_generations import datasets
from gensim.models import Word2Vec, FastText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


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
    if cache is not None and os.path.isfile(cache):
        df = pd.read_pickle(cache)
        print('read dataframe from cache', cache, sep=' ')
        return df
    if out is not None and os.path.isfile(out):
        print('read dataframe from csv file', os.path.join(os.getcwd(), out), sep=' ')
        return pd.read_csv(os.path.join(os.getcwd(), out))
    # log datasets that did not load well, without ending the whole process.
    with open('bad_data_sets.txt', 'w+') as f:
        f.write('bad data sets: \n')
    df = pd.DataFrame()  # Initial dataframe that will be grown
    print('Downloading datasets, expect this to take a few minutes')
    for key in datasets:
        try:
            print('Downloading {}'.format(key))
            download_extract(key)  # Downloads this data set as zip and extracts it.
            print('Merging {}'.format(key))
            new = makeSingleDf(path + '/' + key)  # Merges the 50+- CSVs in this generation to a single df
            # We used to do this later, but it seems that there are too much lives in this new dataset We now will
            # drop them so if there is any finalized version, it will win de-duping (was already not supposed to happen)
            drop_lives(new)
            df = pd.concat([df, new], ignore_index=True, sort=True)
            remove_duplicates(df)
            erase(key)
        except Exception as e:
            with open('bad_data_sets.txt', 'a+') as f:
                f.write(key + '\n')
            erase(key)
            print(e)
            continue
    print('there are ', len(df.index), ' records in data set', sep='')
    # save united dataset to csv or pickle.
    if out is not None:
        df.to_csv(path_or_buf=out, chunksize=10000)
    if cache is not None:
        df.to_pickle(path=cache)
        print('Saved data frame to', cache)
    return df


def makeSingleDf(path):
    chunk_list = []
    for fileName in sorted(os.listdir(path)):  # Sort to be deterministic
        if fileName.endswith('.csv'):
            chunk = pd.read_csv(os.path.join(path, fileName))
            chunk_list.append(chunk)
    print('Read ', len(chunk_list), 'csv files')
    df = pd.concat(chunk_list, ignore_index=True)
    return df


def downloadData(path=r'rawData/'):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for generation in datasets:
        downloadFile(path + generation + '.zip', datasets[generation])
        print('Downloaded', generation, sep=' ')


def downloadFile(fileName, url, path=r'rawData/'):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # fileName = path + fileName
    if os.path.isfile(fileName) or os.path.exists(fileName[:-4]):
        print('file {} already downloaded'.format(fileName))
        return
    with urllib.request.urlopen(url) as response, open(fileName, 'wb+') as out_file:
        shutil.copyfileobj(response, out_file)
    print('downloaded {}'.format(fileName))


def download_extract(generation, path=r'rawData/'):
    downloadFile(path + generation + '.zip', datasets[generation])
    print('extracting')
    extract(path + generation)


def erase(generation, path=r'rawData/'):
    fileName = path + generation + '.zip'
    if os.path.isfile(fileName):
        os.remove(fileName)
    if os.path.exists(path + generation + '/'):
        shutil.rmtree(path + generation + '/')
        print('erased {}'.format(generation))


def extract(fileName):
    directory = fileName + '/'
    if os.path.exists(directory):
        print(directory, ' exists', sep=' ')
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


def extract_creator(df):
    pat = '\A{"id":([0-9]*),'
    ids = df['creator']
    ids = ids.map(lambda x: int(re.search(pat, x).group(1)))
    df['creator'] = ids


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


def drop_lives(df):
    ind = df[df['state'] == 'live'].index
    df.drop(ind, inplace=True)

def fix_state(df):
    drop_lives(df)
    stat = df['state'].map(lambda x: x if x == 'successful' else 'failed')
    df['state'] = stat


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


def download_photos(df, folder='tmp'):
    folder = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print('created folder')
    for url, id in zip(df['photo'], df['id']):
        with urllib.request.urlopen(url) as response, open(os.path.join(folder, str(id) + '.jpg'), 'wb+') as out_file:
            shutil.copyfileobj(response, out_file)


def make_word_embeddings(df):
    print('started nlp')
    if os.path.isfile('./fastText.model'):
        print('started reading pretrained model file')
        model = FastText.load('./fastText.model')
        print('finished file loading')
    else:
        df1 = df[['name', 'blurb']]
        df1['name'] = df1.apply(lambda r: re.sub("[^A-Za-z']", ' ', r['name'].lower()), axis=1)
        df1['blurb'] = df1.apply(lambda r: re.sub("[^A-Za-z']", ' ', str(r['blurb']).lower()), axis=1)
        df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)

        df_clean = pd.DataFrame({'clean': df2})
        sent = [row.split(',') for row in df_clean['clean']]
        model = FastText(sentences=sent, min_count=1,size= 20, window =3, iter=10)
        model.save("fastText.model")

    print('started setting name_nlp column')
    df['name_nlp'] = df.apply(lambda row: sum([model.wv[re.sub("[^A-Za-z']", ' ', s.lower().strip())] for s in (row["name"] + ' ' + str(row['blurb'])).split(' ')]), axis=1)
    print('finished setting name_nlp column')


def set_semantics(df):
    analyser = SentimentIntensityAnalyzer()
    neg = []
    pos = []
    compound = []

    for row in df['blurb'].astype(str):
        score = analyser.polarity_scores(str(row.encode("utf-8")))
        neg.append(score['neg'])
        pos.append(score['pos'])
        compound.append(score['compound'])

    df['blurb_pos'] = pos
    df['blurb_neg'] = neg
    df['blurb_compound'] = compound


def avg_word(sentence):
  words = sentence.split()
  return float(sum(len(word) for word in words))/len(words)


def set_text_statistics(df):
    nltk.download('stopwords')
    df_str = df["blurb"].astype(str)

    stopwords = set(nltk.corpus.stopwords.words('english'))
    df["name_num_words"]  = df["name"].apply(lambda x: len(x.split()))
    df["name_num_chars"]  = df["name"].apply(lambda x: len(x.replace(" ","")))
    df['name_avg_word_length'] = df['name'].apply(lambda x: avg_word(x))
    df["blurb_num_words"] = df_str.apply(lambda x: len(x.split()))
    df["blurb_num_chars"]  = df_str.apply(lambda x: len(x.replace(" ","")))
    df['blurb_avg_word_length'] = df_str.apply(lambda x: avg_word(x))
    df['blurb_stopwords'] = df_str.apply(lambda x: len([x for x in x.split() if x in stopwords]))
    df['name_upper'] = df['name'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    df['blurb_upper'] = df_str.apply(lambda x: len([x for x in x.split() if x.isupper()]))





def files_check():
    print('downloading')
    download_extract('November 2019')
    print('eraseing')
    erase('November 2019')


if __name__ == '__main__':
    make_dataframe(path=r'rawData', cache='rick.pickle')
