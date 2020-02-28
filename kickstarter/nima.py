import json
import os
import shutil
import urllib.request
from urllib.error import HTTPError

import scipy.stats
from tqdm import tqdm

from kickstarter.logger import logger


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
