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
import js2py
import logging
logging.basicConfig(level=logging.DEBUG)
from urllib2 import Request, urlopen, URLError
from datetime import datetime
import psycopg2
import os
import boto


def get_revenue_table(file):
    f = codecs.open(file, 'r',"utf-8")
    soup = BeautifulSoup(f, "html5lib")
    try:
        # Extract movie info from main block
        title1 = unidecode(soup.title.get_text().replace(" - Daily Box Office Results - Box Office Mojo", ""))
        title2 = soup.body.find(id="container").find(id="main").find(id="body").select("table")[2].tbody.tr.td.select("table")[0].tbody.tr.select('td')[1].b.get_text()
        info = soup.body.find(id="container").find(id="main").find(id="body").select("table")[2].tbody.tr.td.select("table")[0].tbody.tr.center.tbody.select("b")
        total_revenues = int(re.sub('[!@#$,]', '', info[0].get_text()))
        distributor = unidecode(re.sub('[!@#$,]', '', info[1].get_text()))
        release_date = unidecode(re.sub('[!@#$,]', '', info[2].get_text()))
        dt_obj = datetime.strptime(release_date, '%B %d %Y')
        # Of the form datetime.datetime(2016, 5, 6, 0, 0) (e.g. dt_obj.year = 2016)
        genre = unidecode(re.sub('[!@#$,]', '', info[3].get_text()))
        runtime_pre = production_budget = re.sub('[!@#$,.a-zA-z]', '', info[4].get_text()).strip().split()
        runtime = int(runtime_pre[0]) * 60 + int(runtime_pre[1])
        MPAA = production_budget = re.sub('[!@#$,]', '', info[5].get_text())

        # Convert production budget string to integer
        production_budget_pre1 = unidecode(re.sub('[!@#$,]', '', info[6].get_text()))
        production_budget_pre2 = "".join(production_budget_pre1.lower().split())
        for word, initial in {"million":"000000", "thousand":"000" }.items():
            production_budget = production_budget_pre2.replace(word.lower(), initial)

        # Extract revenue figures
        java_text = soup.find_all(type="text/javascript")
        t = java_text[5]
        jtext = t.getText().split('\t')[6].replace("\n", " ")
        table = js2py.eval_js(jtext)
        revenue = table.to_list()

        # Enter into dataframe
        rev = pd.DataFrame(revenue)
        rev.drop(0, axis=1, inplace=True)

        # Get total base revenues
        base_revenues = rev[1].sum()

        # Calculate conversion factor
        rev_cf = (total_revenues / base_revenues) / 1000000

        # Load onto dictionary to export
        keys = ['title1', 'title2', 'total_revenues', 'distributor', 'dt_obj', 'genre', 'runtime', 'MPAA', 'production_budget']

        values = [title1, unidecode(title2), total_revenues, distributor, dt_obj, genre, runtime, MPAA, int(production_budget)]

        movie_details = dict(zip(keys, values))

        rev_df = rev * rev_cf

    except Exception, e:
        print file
        movie_details = {'title1': title1}
        rev_df = 0
        missing_information.append(file)
        logging.exception(e)

    return movie_details, rev_df

def franchises():
    pass

def load_df(d, rev_df):
    df2 = pd.DataFrame(d, index=[1,])
    ts = df2['dt_obj'][1]

    season = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}

    columns = ['year', 'month', 'day', 'season']
    column_vals = [ts.year, ts.month, ts.day, season[ts.month]]
    date_data = pd.DataFrame(dict(zip(columns, column_vals)), index=[1,])
    df2.join(date_data)


    return df2

def populate_dataframe(file, df):
    d, rev_df = get_revenue_table(file)
    if len(d) > 1:
        pass
        df2 = load_df(d, rev_df)
        df.append(df2, ignore_index=True)
    else:
        missing_information.append(d['title1'])

    return df


if __name__ == '__main__':
    # dead link example
    # link = "index.html?page=daily&view=chart&id=1dwhereweare.htm&adjust_yr=2016&p=.htm"
    missing_information = []

    link = "index.html?page=daily&view=chart&id=gladiator.htm&adjust_yr=2016&p=.htm"
    d, rev_df = get_revenue_table(link)

    # Connect to S3
    # access_key, access_secret_key = os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY')
    # conn = boto.connect_s3(access_key, access_secret_key)
    # # List all the buckets
    # all_buckets = [b.name for b in conn.get_all_buckets()]
    # print all_buckets

    #
    # # To strip html from critic reviews
    #
    # url = 'http://christylemire.com/captain-america-civil-war/'
    # page = urllib2.urlopen(url)
