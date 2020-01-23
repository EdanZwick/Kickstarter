import json

import shutil
import numpy as np


def extract_creator_fields(df):
    get_registration_stat = lambda row: json.loads(row.creator).get('is_registered', 'unknown')
    df['creator_status'] = df.apply(get_registration_stat, axis=1)
    get_creator_photo = lambda row: json.loads(row.creator).get('avatar').get('medium')
    df['creator_photo'] = df.apply(get_creator_photo, axis=1)
    get_superbacker = lambda row: json.loads(row.creator).get('is_superbacker', 'unkown')
    df['super_creator'] = df.apply(get_superbacker, axis=1)


def get_creator_history(df):
    counts = df['creator id'].value_counts()
    # subtract one as we don't want to count the current project
    get_history = lambda row: counts[row['creator id']] - 1
    df['creator_past_proj'] = df.apply(get_history, axis=1)
    winners = df.loc[df['state'] == 'successful']
    win_counts = winners['creator id'].value_counts()
    df['creator_successes'] = df.apply(_get_success, axis=1, counts=win_counts)
    # Tenerary clause if due to the fact that if the project is successful, it was excluded twice (in total count and in succ. count)
    get_unsuc = lambda row: row['creator_past_proj'] - row['creator_successes'] if row['state'] == 'successful' else \
        row['creator_past_proj'] - row['creator_successes']
    df['creator_unsuccesses'] = df.apply(get_unsuc, axis=1)


def _get_success(row, counts=None):
    try:
        # we subtract the 1 as we don't want to consider the current project twards project's history.
        return counts[row['creator id']] - 1
    except KeyError:
        return 0


def fix_state(df):
    df.drop(df[~df.state.isin(['successful', 'failed'])].index, inplace=True)


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
