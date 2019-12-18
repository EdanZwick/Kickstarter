__author__ = 'amir'

import pandas as pd
import matplotlib.pyplot as plt
import dataCleaning as dc
import visio
import seaborn as sns

import dataCleaning as dc


import knn_model as knn
import logistic_regression_model as logistic
import random_forest_model as forest
import gradient_boosting_model as gradient_boosting

df = dc.make_dataframe()

cols = list(df.columns.values)
print(cols)
num_recs = len(df.index)
print()
print('There are originaly '+ str(num_recs) + ' records in data')

redundant = ['country_displayable_name', 'currency_symbol', 'currency_trailing_code', 'current_currency',
             'source_url','disable_communication', 'profile','urls','photo', 'usd_pledged', 'usd_type']
df.drop(columns=redundant, inplace=True)
print('sanity check, print new columns:')
cols = list(df.columns.values)
print(cols)


nes = df.isna().sum()
print(nes)


empty = ['friends','is_backing','is_starred','permissions']
df.drop(columns=empty,inplace=True)
cols = list(df.columns.values)
print(cols)

timefields = ['created_at','deadline','launched_at','state_changed_at']
dc.convert_time(df,timefields)
print('sanity check')

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df.head())


dc.extract_creator(df) #replaces the creator json with creator id int
dc.extract_catagories(df) #gets project catagory data

#
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df.head())


print('There are originally ' + str(num_recs) + ' records in data')
dc.remove_duplicates(df)
num_recs = len(df.index)
print('After processing there are ' + str(num_recs) + ' records in data')

dc.convert_goal(df)

# extra manipulations

dc.extract_month_and_year(df, timefields)
dc.add_destination_delta_in_months(df)


# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df.head())

# df.to_csv('./cleaned_data.csv')

######visualizations
#
# visio.plot_success_by_category_name(df)
# visio.plot_success_by_sub_category_name(df) #too many sub categories
# visio.plot_success_by_launched_year(df)
# visio.plot_success_by_launched_month(df)
# visio.plot_distribution_by_state(df)
# visio.plot_distribuition_by_state_squarify(df)
# visio.plot_distriubtion_by_state_slice(df)
# visio.plot_success_by_destination_delta_in_months(df)

#filter to only deal with successful and failed projects and set state to {0,1} values

#preparing to train
df = df.loc[df['state'].isin(['failed','successful'])]
dc.encode_string_enums(df, 'state', ['failed','successful'], [0,1])
# dc.make_word_embeddings(df)


dc.set_semantics(df)
dc.set_text_statistics(df)
#models

knn.run_model(df)  #73 precision and 69 semantic parsing
logistic.run_model(df) #67.8 precision and 69 semantic parsing and 69 semantic parsing
forest.run_model(df) #75 percision and 77 semantic parsing
gradient_boosting.run_model(df) #77 and 79 semantic parsing