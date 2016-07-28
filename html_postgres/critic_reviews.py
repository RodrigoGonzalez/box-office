import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
from unidecode import unidecode
import re
import urllib2
import requests

def load_dataframe():
    df = pd.read_csv('movie_revs.csv')
    df['key'] = ["".join([re.sub('[!@#$/:,]', '', x) for x in y.lower().split()]) for y in df['title2']]
    return df

# def 

if __name__ == '__main__':
    df = load_dataframe()
    columns = ['key', 'critic', 'review']
    df_reviews = pd.DataFrame(columns=columns)
