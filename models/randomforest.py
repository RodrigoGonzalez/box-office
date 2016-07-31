import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import yahoo_finance
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score, train_test_split

def process_genres(string):
    """
    INPUT: string
    OUTPUT:
    """
    return "/".join(string.replace("/", "").split())

def process_distributor(string):
    """
    INPUT: string
    OUTPUT:
    """
    return "/".join([x.strip() for x in string.split("/")])

def process_date(string):
    """
    INPUT: string
    OUTPUT:
    """
    date = string.split('-')
    date[2] = '01'
    return '-'.join(date)

def market_date(string, ticker):
    """
    INPUT: string, ticker representation
    OUTPUT: float
    """
    date = datetime.strptime(string, '%Y-%m-%d')
    delta = timedelta(days=2)
    delta0 = (date - delta).strftime('%Y-%m-%d')
    delta1 = (date + delta).strftime('%Y-%m-%d')
    historical_close = ticker.get_historical(delta0, delta1)
    close = np.mean([float(x['Close']) for x in historical_close])
    return close

def release_info(release_dates):
    """
    INPUT: TIME DATA FRAME
    OUTPUT: TIME FEATURES
    """
    df_ts = pd.DataFrame()

    df_CPI_entertainment = pd.read_csv('../data/CPI-UrbanConsumers-AdmissionMoviesTheatersConcerts.csv')
    df_CPI = pd.read_csv('../data/CPIAUCSL.csv')

    CPI = dict(zip(df_CPI['DATE'], df_CPI['CPIAUCSL']))
    CPI_E = dict(zip(df_CPI_entertainment['DATE'], df_CPI_entertainment['CUSR0000SS62031']))

    S = yahoo_finance.Share('^GSPC')
    N = yahoo_finance.Share('^IXIC')

    df_ts['keys'] = release_dates['dt_obj'].apply(process_date)
    df_ts['CPI'] = [CPI[x] for x in df_ts['keys']]
    df_ts['CPI_E'] = [CPI_E[x] for x in df_ts['keys']]
    df_ts['SP500'] = [market_date(x, S) for x in release_dates['dt_obj']]
    df_ts['NASDAQ'] = [market_date(x, N) for x in release_dates['dt_obj']]
    df_ts.to_csv('market_data.csv')
    return df_ts

def load_dataframe():
    """
    INPUT: filename preprocessed revenue data
    OUTPUT: dataframe ready for modelling
    """

    df = pd.read_csv('../html_postgres/movie_revs.csv')
    df = df.query('year > 2005 and year < 2016').reset_index()
    df['distributor'] = df['distributor'].apply(process_distributor)
    df['genre'] = df['genre'].apply(process_genres)


    distributors = df.pop('distributor').str.get_dummies(sep="/")
    distributors.drop(labels='A24', inplace=True, axis=1)
    genres = df.pop('genre').str.get_dummies(sep="/")
    genres.drop(labels='Action', inplace=True, axis=1)

    time_columns = ['dt_obj', 'year', 'month', 'day']
    # release_dates = df[time_columns]
    # df_ts = release_info(df['time_columns'])
    df_ts = pd.load_csv('market_data.csv')
    df = df.join(distributors)
    df = df.join(genres)
    df = df.join(df_ts)

    columns_to_dummy = ['MPAA', 'season', 'month']
    df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)

    columns_to_drop = ['Unnamed: 0', 'day', 'title1', 'title2','dt_obj', 'index', 'keys']
    df.drop(labels=columns_to_drop, inplace=True, axis=1)

    return df

def RFR():
    df = load_dataframe()
    y = df.pop('total_revenues').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_fitted = rf.fit(X_train,y_train)
    y_predict = rf_fitted.predict(X_test)
    score = cross_val_score(rf, X, y, scoring='r2', cv=3).mean()

    return rf

if __name__ == '__main__':
    rf = RFR()
    print("Score with the entire dataset = %.2f" % score)

# scoring
# ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
