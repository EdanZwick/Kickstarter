import json
import re

import shutil
from datetime import datetime

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
    # escape_quotes = lambda row: re.sub(r'([^:,{}])(")([^:,{}])',r'\1\"\3',row.creator)
    escape_quotes = lambda row: re.sub(r'("[^"]*":)("[^"\\]*)(")([^"]*)(")([^"\\]*",)', r'\1\2\"\4\"\6', row.creator)
    df['creator'] = df.apply(escape_quotes, axis=1)
    escape_uneven_quotes = lambda row: re.sub(r'([A-Za-z])"([A-Za-z])', r'\1\"\2', row.creator)
    df['creator'] = df.apply(escape_uneven_quotes, axis=1)
    # parse creator id out of the now legal json
    df['creator_id'] = df.apply(get_creator_id, axis=1)


# seperated to enable error handling
def get_creator_id(row):
    try:
        return json.loads(row['creator'])['id']
    except:
        return -1


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


def get_image_url(df):
    imgs = df['photo']
    imgs = imgs.apply(json.loads)
    imgs = imgs.map(lambda x: x.get('full', 0))
    df['photo'] = imgs


def extract_creator_fields(df):
    get_registration_stat = lambda row: json.loads(row.creator).get('is_registered', 'unknown')
    df['creator_status'] = df.apply(get_registration_stat, axis=1)
    get_creator_photo = lambda row: json.loads(row.creator).get('avatar').get('medium')
    df['creator_photo'] = df.apply(get_creator_photo, axis=1)
    get_superbacker = lambda row: json.loads(row.creator).get('is_superbacker', 'unkown')
    df['super_creator'] = df.apply(get_superbacker, axis=1)
