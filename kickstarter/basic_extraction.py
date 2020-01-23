import json
import re
from datetime import datetime


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