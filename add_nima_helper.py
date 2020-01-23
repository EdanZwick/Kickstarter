import kickstarter.nima
from kickstarter.data_loader import get_pickles

def _download_nima_pickle():
    df = get_pickles('with_NIMA.pickle')
    kickstarter.nima.add_nima(df, jsonFile='NIMA predictions/predictions_imgs_all_technical.json', columnName='nima_tech')
    df.to_pickle('pickled_data/with_NIMA.pickle')


_download_nima_pickle()
