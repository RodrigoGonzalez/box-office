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

missing_information = []

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
        production_budget = re.sub('[!@#$,]', '', info[6].get_text())

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

        labels = ['title1', 'title2', 'total_revenues', 'distributor', 'dt_obj', 'genre', 'runtime', 'MPAA', 'production_budget', 'rev', 'rev_cf']
    except Exception, e:
        print file
        missing_information.append(file)
        logging.exception(e)

    return title2, rev



if __name__ == '__main__':
    link = "index.html?page=daily&view=chart&id=marvel2016.htm&adjust_yr=2016&p=.htm"

    title, tot_rev = get_revenue_table(link)


    #
    # # To strip html from critic reviews
    #
    # url = 'http://christylemire.com/captain-america-civil-war/'
    # page = urllib2.urlopen(url)
