# Based on "Yahoo Finance (hidden) API"
# https://greenido.wordpress.com/2009/12/22/work-like-a-pro-with-yahoo-finance-hidden-api/

import urllib

"""

This module provides a Python API for retrieving stock data from Yahoo Finance.
sample usage:
>>> import yahoo_quotes
>>> print yahoo_quotes.get_price('GOOG')
830.76
"""


def __request(symbol, stat):
    url = f'http://finance.yahoo.com/d/quotes.csv?s={symbol}&f={stat}'
    return urllib.urlopen(url).read().strip().strip('"')

def get_all(symbol):
    """
    Get all available quote data for the given ticker symbol.

    Returns a dictionary.
    """
    values = __request(symbol, 'l1c1va2xj1b4j4dyekjm3m4rr5p5p6s7n').split(',')
    return {
        'price': values[0],
        'change': values[1],
        'volume': values[2],
        'avg_daily_volume': values[3],
        'stock_exchange': values[4],
        'market_cap': values[5],
        'book_value': values[6],
        'ebitda': values[7],
        'dividend_per_share': values[8],
        'dividend_yield': values[9],
        'earnings_per_share': values[10],
        '52_week_high': values[11],
        '52_week_low': values[12],
        '50day_moving_avg': values[13],
        '200day_moving_avg': values[14],
        'price_earnings_ratio': values[15],
        'price_earnings_growth_ratio': values[16],
        'price_sales_ratio': values[17],
        'price_book_ratio': values[18],
        'short_ratio': values[19],
        'name': values[20],
    }


def get_price(symbol):
    return __request(symbol, 'l1')


def get_change(symbol):
    return __request(symbol, 'c1')


def get_volume(symbol):
    return __request(symbol, 'v')


def get_avg_daily_volume(symbol):
    return __request(symbol, 'a2')


def get_stock_exchange(symbol):
    return __request(symbol, 'x')


def get_market_cap(symbol):
    return __request(symbol, 'j1')


def get_book_value(symbol):
    return __request(symbol, 'b4')


def get_ebitda(symbol):
    return __request(symbol, 'j4')


def get_dividend_per_share(symbol):
    return __request(symbol, 'd')


def get_dividend_yield(symbol):
    return __request(symbol, 'y')


def get_earnings_per_share(symbol):
    return __request(symbol, 'e')


def get_52_week_high(symbol):
    return __request(symbol, 'k')


def get_52_week_low(symbol):
    return __request(symbol, 'j')


def get_50day_moving_avg(symbol):
    return __request(symbol, 'm3')


def get_200day_moving_avg(symbol):
    return __request(symbol, 'm4')


def get_price_earnings_ratio(symbol):
    return __request(symbol, 'r')


def get_price_earnings_growth_ratio(symbol):
    return __request(symbol, 'r5')


def get_price_sales_ratio(symbol):
    return __request(symbol, 'p5')


def get_price_book_ratio(symbol):
    return __request(symbol, 'p6')


def get_short_ratio(symbol):
    return __request(symbol, 's7')


def get_historical_prices(symbol, start_date, end_date):
    """
    Get historical prices for the given ticker symbol.
    Date format is 'YYYYMMDD'

    Returns a nested list.
    """
    url = (
        (
            (
                'http://ichart.yahoo.com/table.csv?s={0}&'.format(symbol)
                + 'd={0}&'.format(str(int(end_date[4:6]) - 1))
                + 'e={0}&'.format(str(int(end_date[6:8])))
                + 'f={0}&'.format(str(int(end_date[:4])))
            )
            + 'g=d&'
        )
        + 'a={0}&'.format(str(int(start_date[4:6]) - 1))
        + 'b={0}&'.format(str(int(start_date[6:8])))
        + 'c={0}&'.format(str(int(start_date[:4])))
        + 'ignore=.csv'
    )
    days = urllib.urlopen(url).readlines()
    data = [day[:-2].split(',') for day in days][1:]
    fields = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose']
    return fields, data
