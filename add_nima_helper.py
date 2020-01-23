import dataCleaning as dc
import kickstarter.data_loader

df = kickstarter.data_loader.get_pickles('with_NIMA.pickle')
dc.add_nima(df, jsonFile='NIMA predictions/predictions_imgs_all_technical.json', columnName = 'nima_tech')
df.to_pickle('pickled_data/with_NIMA.pickle')
