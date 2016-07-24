import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
import urllib2
import string
import re
import csv
from unidecode import unidecode
import logging
logging.basicConfig(level=logging.DEBUG)
from urllib2 import Request, urlopen, URLError

def get_link_list(file, df, columns):
    f = codecs.open(file, 'r',"utf-8")
    soup = BeautifulSoup(f, "html5lib")
    rows = soup.table.select('tr')
    rows.pop(0)

    franchise_links = []
    for row in rows:
        link = unidecode(row.a['href'])
        row_text = unidecode(row.get_text()).split('\n')
        name = row_text.pop(1)
        figures = [int(re.sub('[!@#$,]', '',x)) for x in row_text[1:-2]]
        franchise_links.append(link)
        figures.append(name)
        figures.append(link)
        df2 = pd.DataFrame(dict(zip(columns, figures)), index=[1,])
        df = df.append(df2, ignore_index=True)
    return franchise_links, df

def extract_franchise_names(franchise_links):
    base_url = ''
    for link in franchise_links:
        url = ("http://www.the-numbers.com/" + link + "#tab=summary")
        print url
        try:
            page = urllib2.urlopen(url)
            soup = BeautifulSoup(page, "html5lib")
            print soup

        except Exception, e:
            logging.exception(e)
        break

if __name__ == '__main__':
    file = 'franchises_html.txt'
    columns = ['num', 'domestic', 'adj_domestic', 'worldwide', 'first_year', 'last_year', 'franchise', 'link']
    df = pd.DataFrame()
    links, df = get_link_list(file, df, columns)
    extract_franchise_names(links)
