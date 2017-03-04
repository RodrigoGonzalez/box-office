import os, re
import pandas as pd
import numpy as np
from lxml import html
import cPickle as pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm

def load_dataframe():
    """

    """
    df = pd.read_csv('../html_postgres/movie_revs.csv')
    df['movie_key'] = ["".join([re.sub('[!@#$/:,]', '', x) for x in y.lower().split()]) for y in df['title2']]
    df['match_key'] = df['title1'].apply(word_set)
    return df

def word_set(string):
    """
    INPUT: string
        - stripped and clean title2
    Returns cleaned movie_key
    """
    return set(re.sub('\W', ' ', string).lower().split())
    # for flagging

columns = ['production_budget', 'runtime', 'month', 'season', 'total_revenues']


df_ = load_dataframe()
df_['tot_revs'] = np.log1p(df_.total_revenues)
df_['prod_bud'] = np.log1p(df_.production_budget)


'Unnamed: 0', u'MPAA', u'distributor', u'dt_obj', u'genre',
       u'production_budget', u'runtime', u'title1', u'title2',
       u'total_revenues', u'day', u'month', u'season', u'sleeper', u'year',
       u'movie_key', u'match_key', u'tot_revs', u'prod_bud'],
      dtype='object')

Revenues
x = df_['total_revenues'].get_values()
b = sns.distplot(x, kde=True, rug=True, hist=True);
b.axes.set_title('Total Revenues Distribution')
sns.despine(offset=10, trim=True)
plt.ylabel('$ USD B')
plt.show()


b = sns.boxplot(x='month', y='total_revenues', data=df_, palette="Blues") # hue="sex",
b.axes.set_title('Boxplot of Month Released vs Total Revenues')
sns.despine(offset=10, trim=True)
plt.xlabel('Month Released')
plt.ylabel('$ USD B')
plt.show()

iris = df_[columns]
sns.pairplot(iris);
plt.show(False)


b = sns.boxplot(x=df_['MPAA'], y=df_['total_revenues'], palette="Blues") # hue="sex",
b.axes.set_title('Boxplot of MPAA Rating vs Total Revenues')
sns.despine(offset=10, trim=True)
plt.xlabel('MPAA Rating')
plt.ylabel('$ USD B')
plt.show(False)

# Draw a plot of two variables with bivariate and univariate graphs.
######################################################

g = sns.jointplot("production_budget", "total_revenues", data=df_, kind="reg")
g.fig.suptitle('Production Budget vs Total Revenues')
sns.despine(offset=10, trim=True)
plt.xlabel('Production Budget (USD B)')
plt.ylabel('Total Revenues (USD B)')
plt.show(False)


g = sns.jointplot("prod_bud", "tot_revs", data=df_, kind="reg")
g.fig.suptitle('Natural log Transforms of Production Budget vs Total Revenues')
sns.despine(offset=10, trim=True)
plt.xlabel('ln(Production Budget)')
plt.ylabel('ln(Total Revenues)')
plt.show(False)

g = sns.jointplot("production_budget", "tot_revs", data=df_, kind="reg")
g.fig.suptitle('Production Budget vs Total Revenues')
sns.despine(offset=10, trim=True)
plt.xlabel('Production Budget (USD B)')
plt.ylabel('Total Revenues (USD B)')
plt.show(False)

g = sns.jointplot("prod_bud", "total_revenues", data=df_, kind="reg")
g.fig.suptitle('Natural log Transforms of Production Budget vs Total Revenues')
sns.despine(offset=10, trim=True)
plt.xlabel('ln(Production Budget)')
plt.ylabel('ln(Total Revenues)')
plt.show(False)



df_ = load_dataframe()
df_
df_.total_revenues.max()
df_.total_revenues.min()
df_.describe()
df_.describe().transpose()
skew = df_.skew()
kurt = df_.kurtosis()
stats_frame = pd.concat([stats, skew, kurt], axis=1)
stats_frame.rename(columns={0: 'skew', 1: 'kurtosis'}, inplace=True)
stats = df_.describe().transpose()
stats_frame.rename(columns={0: 'skew', 1: 'kurtosis'}, inplace=True)


df_ = load_dataframe()
df_
df_.total_revenues.max()
df_.total_revenues.min()

import math

df_['tot_revs'] = np.log1p(df_.total_revenues)
df_
df_['prod_bud'] = np.log1p(df_.production_budget)
df_.columns
