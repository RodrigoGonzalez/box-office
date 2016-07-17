import pandas as pd
import numpy as np
import omdb
from pymongo import MongoClient

def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print 'WARNING', response.status_code
    else:
        return response.json()

def main():
    # include full plot and Rotten Tomatoes data
    a = omdb.get(title='True Grit', year=1969, fullplot=True, tomatoes=True)
    # set timeout of 5 seconds for this request
    b = omdb.get(title='True Grit', year=1969, fullplot=True, tomatoes=True, timeout=5)
    return a, b

if __name__ == '__main__':
    a, b = main()
    # link = 'http://www.omdbapi.com/?'
    # # payload = {'api-key': 'YOUR API KEY HERE'}
    # html_str = single_query(link) # , payload)
