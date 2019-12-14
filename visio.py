import matplotlib.pyplot as plt
import dataCleaning as dc
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
import squarify

def plot_projStat(df):
    counts = df['state'].value_counts(sort=False)
    counts.plot.bar(x='State',y='count',title='Number of projects by status')

def plot_goaldist(df):
    sns.distplot(df['goal'])


def plot_success_by_category_name(df):
    suc = df.loc[df['state'] == 'successful']
    f = df.loc[df['state'] == 'failed']
    sc = suc['parent_category_name'].value_counts()
    sf = f['parent_category_name'].value_counts()
    d = pd.DataFrame({'Successful': sc, 'failed': sf})
    d.plot.bar()
    plt.show()

def plot_success_by_sub_category_name(df):
    suc = df.loc[df['state'] == 'successful']
    f = df.loc[df['state'] == 'failed']
    sc = suc['category_name'].value_counts()
    sf = f['category_name'].value_counts()
    d = pd.DataFrame({'Successful': sc, 'failed': sf})
    d.plot.bar()
    plt.show()

def plot_success_by_launched_year(df):
    suc = df.loc[df['state'] == 'successful']
    f = df.loc[df['state'] == 'failed']
    sc = suc['launched_at_year'].value_counts()
    sf = f['launched_at_year'].value_counts()
    d = pd.DataFrame({'Successful': sc, 'failed': sf})
    d.plot.bar()
    plt.show()

def plot_success_by_launched_month(df):
    suc = df.loc[df['state'] == 'successful']
    f = df.loc[df['state'] == 'failed']
    sc = suc['launched_at_month'].value_counts()
    sf = f['launched_at_month'].value_counts()
    d = pd.DataFrame({'Successful': sc, 'failed': sf})
    d.plot.bar()
    plt.show()


def plot_distribution_by_state(df):
    state = round(df["state"].value_counts() / len(df["state"]) * 100,2)
    labels = list(state.index)
    values = list(state.values)

    trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))
    layout = go.Layout(title='Distribuition of States', legend=dict(orientation="h"));
    fig = go.Figure(data=[trace1], layout=layout)
    iplot(fig)

def plot_distribuition_by_state_squarify(df):
    state = df['state'].value_counts()
    successful = state['successful']
    failed = state['failed']
    total = state.values.sum()
    others = total - failed - successful

    squarify.plot(sizes=[successful,failed, others],
              label=["Failed ("+str(round(float(failed)/total * 100.,2))+"%)",
                     "Successful ("+str(round(float(successful)/total * 100.,2))+"%)",
                     "Others ("+str(round(float(others)/total * 100.,2))+"%)"], color=["blue","red","green"], alpha=.4 )
    plt.title('State', fontsize = 20)
    plt.axis('off')
    plt.show()

def plot_distriubtion_by_state_slice(df):
    plt.style.use('seaborn-pastel')

    fig, ax = plt.subplots(1, 1, dpi=100)
    explode = [0,0,.1,.2, .4]
    df.state.value_counts().plot.pie(autopct='%0.2f%%', explode=explode)
    plt.title('Breakdown of Kickstarter Project Status')
    plt.ylabel('')
    plt.show()


def plot_success_by_destination_delta_in_months(df):
    suc = df.loc[df['state'] == 'successful']
    f = df.loc[df['state'] == 'failed']
    sc = suc['destination_delta_in_months'].value_counts()
    sf = f['destination_delta_in_months'].value_counts()
    d = pd.DataFrame({'Successful': sc, 'failed': sf})
    d.plot.bar()
    plt.show()

if __name__ == '__main__':
       df = dc.make_dataframe()
       plot_projStat(df)