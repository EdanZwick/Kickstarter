import json
import os
import shutil
import urllib.request
from urllib.error import HTTPError

import scipy.stats
from tqdm import tqdm

from kickstarter.logger import logger


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
    return score
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


def download_photos(df, url_column='photo', name_column='id', folder='tmp'):
    folder = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info('created folder')
    for i, (url, idnum) in enumerate(zip(df[url_column], df[name_column])):
        if (i % 10000 == 0):
            logger.info('downloaded {} images'.format(i))
        try:
            with urllib.request.urlopen(url) as response, open(os.path.join(folder, str(idnum)), 'wb+') as out_file:
                shutil.copyfileobj(response, out_file)
        except HTTPError as err:
            with open('bad_images.txt', 'a+') as f:
                f.write(str(idnum) + '\n')
                continue
    logger.info('Downloaded {} images'.format(str(len(df))))

# if __name__ == '__main__':
#     df = get_pickles('creators.pickle')
#     download_photos(df, url_column='creator photo', name_column='creator id', folder='creator photos')
