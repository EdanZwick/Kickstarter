import json

import shutil
import numpy as np

from kickstarter.data import Data


def fix_state(data: Data):
    wanted_labels = data.le.transform(['successful', 'failed'])

    train_index_to_remove = data.train_y[~data.train_y.isin(wanted_labels)].index
    test_index_to_remove = data.test_y[~data.test_y.isin(wanted_labels)].index

    data.train_df.drop(train_index_to_remove, inplace=True)
    data.train_y.drop(train_index_to_remove, inplace=True)

    data.test_df.drop(test_index_to_remove, inplace=True)
    data.test_y.drop(test_index_to_remove, inplace=True)


def extract_country(df):
    # handle cases where there is no location field (just take from country)
    fix_nan = lambda row: '{{"country" : "{}", "state":"unknown"}}'.format(row['country']) if (
            isinstance(row.location, float) and np.isnan(row.location)) else row['location']
    df['location'] = df.apply(lambda row: fix_nan(row), axis=1)
    # Extract the country column from json
    country_from_json = lambda j: json.loads(j)['country']
    df['country'] = df.apply(lambda row: country_from_json(row['location']), axis=1)


def unify_countries(df, counts, threshold):
    unify_less_frequent = lambda row: 'Global' if counts[row['country']] <= threshold else row['country']
    df['country'] = df.apply(lambda row: unify_less_frequent(row), axis=1)


def get_us_state(df):
    states = set(
        ['WA', 'DE', 'DC', 'WI', 'WV', 'HI', 'FL', 'FM', 'WY', 'NH', 'NJ', 'NM', 'TX', 'LA', 'NC', 'ND', 'NE', 'TN',
         'NY', 'PA', ' V', 'CA', 'NV', 'VA', 'GU', 'CO', 'PW', 'AK', 'AL', 'AS', 'AR', 'VT', 'IL', 'GA', 'IN', 'IA',
         'OK', 'AZ', 'ID', 'CT', 'ME', 'MD', 'MA', 'OH', 'UT', 'MO', 'MN', 'MI', 'MH', 'RI', 'KS', 'MT', 'MP', 'MS',
         'PR', 'SC', 'KY', 'OR', 'SD'])
    fix_nan = lambda row: '{"state":"unknown"}' if (
            isinstance(row.location, float) and np.isnan(row.location) and row.country == 'US') else row.location
    df['location'] = df.apply(lambda row: fix_nan(row), axis=1)
    state_from_json = lambda j: json.loads(j)['state'] if json.loads(j)['state'] is not None and json.loads(j)[
        'state'] in states else 'unknown'
    df['country'] = df.apply(
        lambda row: 'US ' + state_from_json(row['location']) if row['country'] == 'US' else row['country'], axis=1)


def encode_string_enums(df, col, str_values, number_values):
    mapping = {str_val: number_val for str_val, number_val in zip(str_values, number_values)}
    df = df[col] = df[col].map(mapping)


def add_destination_delta_in_days(df):
    delta = lambda r: (r['deadline'] - r['launched_at']).components.days
    df['destination_delta_in_days'] = df.apply(lambda row: delta(row), axis=1)


def erase_photos(folder):
    if folder is None:
        raise ValueError('No path to delete')
    shutil.rmtree(folder)
