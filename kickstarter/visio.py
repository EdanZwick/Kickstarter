import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import urllib.request
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
import squarify

plt.style.use('seaborn')


def plot_success_failure(df, col, ax=None):
    plt.style.use('seaborn')
    if ax is None:
        _, ax = plt.subplots(1, 1)
    suc = df.loc[df['state'] == 'successful']
    f = df.loc[df['state'] == 'failed']
    sc = suc[col].value_counts()
    sf = f[col].value_counts()
    d = pd.DataFrame({'Successful': sc, 'failed': sf})
    d.plot(ax=ax, kind='bar',figsize=[20, 10])


def plot_success_by_country(df):
    plot_success_failure(df, 'country')
    plt.title('Success by origin country')
    plt.show()


def plot_success_by_category(df):
    plot_success_failure(df, 'parent_category_name')
    plt.title('Success by parent category')
    plt.show()


def plot_success_by_sub_category(df):
    parents = df.parent_category_name.unique()
    for i, category in enumerate(parents):
        if i % 3 == 0:
            fig, axes = plt.subplots(figsize=[20, 10], nrows=1, ncols=3, sharey='row')
        subpd = df.loc[df['parent_category_name'] == category]
        plot_success_failure(subpd, 'category_name', ax=axes[(i % 3)])
        axes[(i % 3)].set_title(category)
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
    plot_success_failure(df, 'launched_at_month')
    plt.title('Success by the month of campaign launch')
    plt.show()

    
def plot_sccess_by_creator_history(df):
    plot_success_failure(df, 'creator_past_proj')
    plt.title('Success by number of creator\'s past projects, excluding current')
    plt.show()
    plot_success_failure(df, 'creator_successes')
    plt.title('Success by number of creator\'s succesfull past projects, excluding current')
    plt.show()
    plot_success_failure(df, 'creator_unsuccesses')
    plt.title('Success by number of creator\'s un-succesfull past projects, excluding current')
    plt.show()

def plot_distribution_by_state(df):
    state = round(df["state"].value_counts() / len(df["state"]) * 100, 2)
    labels = list(state.index)
    values = list(state.values)
    trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))
    layout = go.Layout(title='Distribuition of States', legend=dict(orientation="h"));
    fig = go.Figure(data=[trace1], layout=layout)
    iplot(fig)


def plot_distriubtion_by_state_slice(df, explode=[0, 0, .1, .2, .4]):
    plt.style.use(['seaborn-pastel'])
    fig, ax = plt.subplots(1, 1, dpi=100)
    fig.set_facecolor('xkcd:pale grey')
    counts = df.state.value_counts()
    counts.plot.pie(autopct='%0.2f%%', explode=explode[:len(counts)])
    plt.title('Breakdown of Kickstarter Project Status')
    plt.ylabel('')
    plt.show()


def plot_success_by_destination_delta_in_days(df):
    data = df.loc[(df['destination_delta_in_days']>14) & (df['destination_delta_in_days'] < 45)]
    plot_success_failure(data, 'destination_delta_in_days')
    plt.title('Success by destination_delta_in_day')
    plt.show()

def plot_distribuition_by_state_squarify(df):
    state = df['state'].value_counts()
    successful = state['successful']
    failed = state['failed']
    total = state.values.sum()
    others = total - failed - successful + 0.001

    squarify.plot(sizes=[successful, failed, others],
                  label=["Failed (" + str(round(float(failed) / total * 100., 2)) + "%)",
                         "Successful (" + str(round(float(successful) / total * 100., 2)) + "%)",
                         "Others (" + str(round(float(others) / total * 100., 2)) + "%)"],
                  color=["blue", "red", "green"], alpha=.4)
    plt.title('State', fontsize=20)
    plt.axis('off')
    plt.show()


def plot_precision(results):
    names = sorted([name for name in results])
    precision = [results[name] for name in names]
    sns.barplot(x=names, y=precision)
    plt.title('Precision by model')
    plt.xticks(rotation=90)
    plt.show()


def plot_success_over_time(df):
    data = df[['launched_at_month', 'launched_at_year', 'state']]
    years = sorted(data['launched_at_year'].unique())
    months = sorted(data['launched_at_month'].unique())
    statistics = []
    for year in years:
        for month in months:
            tmp = data.loc[(data['launched_at_month']==month) & (data['launched_at_year']==year)]
            suc = len(tmp.loc[tmp['state'] == 'successful'])
            total = len(tmp)
            if (total != 0):
                statistics.append(((suc / len(tmp)),(str(month) + '/' + str(year))))
    bars = [name[1] for name in statistics]
    hights = [data[0] for data in statistics]
    chart = sns.barplot(x=bars, y=hights)
    labels = [name if i%6==0 else '' for i, name in enumerate(bars)]
    chart.set_xticklabels(labels = labels, rotation=45)
    plt.title('Success rate by month')
    plt.show()


# Samples 
def display_imgs(df, score_column = 'nima_score', photo_column = 'photo'):
    # sort dataset for easy access to edges.
    ranks = df.sort_values(by=score_column, ascending=False, inplace=False)
    # pick 9 random numbers
    picks = np.random.randint(0,200,9)
    f, axarr = plt.subplots(3,3,figsize=[20, 10])
    # get images from url, and desplay their rating and status
    for i, pic in enumerate(picks):
        img = urllib.request.urlopen(ranks.iloc[pic][photo_column])
        a = plt.imread(img,0)
        axarr[i//3,i%3].imshow(a)
        axarr[i//3,i%3].set_title('NIMA score: {:.2f}    state: {}'.format(ranks.iloc[pic][score_column], ranks.iloc[pic].state))
    # remove plt junk
    for ax in f.axes:
        ax.axis("off")
    plt.rcParams["axes.grid"] = False
    f.suptitle('Random images with high NIMA score',fontsize=40)
    plt.show()
    # again for the failed, using negative index retrieves from the end.
    picks = np.random.randint(0,200,9) * -1
    f, axarr = plt.subplots(3,3,figsize=[20, 10])
    for i, pic in enumerate(picks):        
        img = urllib.request.urlopen(ranks.iloc[pic][photo_column])
        a = plt.imread(img,0)
        axarr[i//3,i%3].imshow(a)
        axarr[i//3,i%3].set_title('NIMA score: {:.2f}    state: {}'.format(ranks.iloc[pic][score_column], ranks.iloc[pic].state))
    for ax in f.axes:
        ax.axis("off")
    plt.rcParams["axes.grid"] = False
    f.suptitle('Random images with low NIMA score',fontsize=40)
    plt.show()
    
    
if __name__ == '__main__':
    df = pd.read_pickle('tmp.pickle')
    # plot_success_by_country(df)
    # plot_distriubtion_by_state_slice(df, explode=[0, 0])
    # plot_success_by_category(df)
    # plot_success_by_sub_category(df)
    # print(plt.style.available)
    # plot_success_by_sub_category_name(df)
    #plot_success_over_time(df)
    plot_success_by_destination_delta_in_days(df)