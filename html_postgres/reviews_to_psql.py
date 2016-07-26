import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
import urllib2
import string
import cPickle as pickle
import re
import time
import csv
from unidecode import unidecode
import logging
logging.basicConfig(level=logging.DEBUG)
from urllib2 import Request, urlopen, URLError
from datetime import datetime

conn = psycopg2.connect(connection)
cursor = conn.cursor()
items = pickle.load(open(pickle_file,"rb"))

for item in items:
    city = item[0]
    price = item[1]
    info = item[2]

    query =  "INSERT INTO items (info, city, price) VALUES (%s, %s, %s);"
    data = (info, city, price)

    cursor.execute(query, data)

conn.commit()
