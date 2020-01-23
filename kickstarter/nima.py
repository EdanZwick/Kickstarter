import json

import scipy.stats

from kickstarter.logger import logger


def add_nima(df, jsonFile, columnName, image_name_is_project='id'):
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
                df.loc[df[image_name_is_project] == iid, columnName] = score
                if i % 10000 == 0:
                    logger.info('done with record {}'.format(i))
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
