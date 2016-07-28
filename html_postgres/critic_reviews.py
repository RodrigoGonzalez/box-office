import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
from unidecode import unidecode
import re
import urllib2
import requests


df = pd.read_csv('movie_revs.csv')
df['key'] = ["".join([re.sub('[!@#$/:,]', '', x) for x in y.lower().split()]) for y in df['title2']]


df = pd.DataFrame()
