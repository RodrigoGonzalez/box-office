import csv, re
import pandas as pd
import numpy as np
from itertools import chain

import yahoo_quotes

def get_distributors(file_name):
    """
    INPUT: string
        - File name of the file containg movie information
    Returns list of distribution companies
    """
    df = pd.read_csv(file_name)
    lst_dist = df.distributor.tolist()

    dists = process_lists(lst_dist)

    with open('distributors.txt', 'wb') as f:
        for d in dists:
            f.write(d + ',\n')

    return dists

def dist_quotes():
    """
    INPUT: list
        - List of the distribution companies
    Return list of stock quotes of media properties
    """
    df = pd.read_csv('distributors_ticker.txt', header=None)

    tickers = df[1].tolist()
    return process_lists(tickers)

def process_lists(lst):
    """
    INPUT: list
        - List of strings, with some seprated by "/", where they need to be separated
    Returns a list of unique strings
    """
    lsts = set(lst)

    l_sep = filter(lambda x: '/' in x, lsts)
    l_clean = [str(x).split('/') for x in l_sep]
    l_add = map(lambda x: x.strip(), list(chain.from_iterable(l_clean)))

    lsts.difference_update(l_sep)
    lsts.update(l_add)
    lsts = sorted(lsts)
    return lsts

def save_quotes(ticks):
    """
    INPUT: list
        - Takes the ticker symbols of the media properties and saves time series s
    Returns a data frame
    """
    start_date, end_date = '20060101', '20160601'

    data_dir = '../data/stock_data/'

    for symbol in ticks:
        hist_data = yahoo_quotes.get_historical_prices(symbol, start_date, end_date)
        stck_data = np.asarray(hist_data)
        df = pd.DataFrame(stck_data[1], columns=stck_data[0])
        df.iloc[:,1:] = df.iloc[:,1:].astype(float)
        path = f"{data_dir}{symbol}.csv"
        df.to_csv(path, index=False)

    return

if __name__ == '__main__':

    # Get distributor names
    file_name = '../html_postgres/movie_revs.csv'
    dists = get_distributors(file_name)

    # Get stock quotes for time series analysis
    media_properties = dist_quotes()
    media_properties.remove('private')
    media_properties.remove('SGX')
    media_properties.remove('HHSE')

    # Save historical stock market data
    save_quotes(media_properties)
