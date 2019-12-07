import matplotlib.pyplot as plt
import dataCleaning as dc
import seaborn as sns

def plot_projStat(df):
    counts = df['state'].value_counts(sort=False)
    counts.plot.bar(x='State',y='count',title='Number of projects by status')

def plot_goaldist(df):
    sns.distplot(df['goal'])



if __name__ == '__main__':
       df = dc.make_dataframe()
       plot_projStat(df)