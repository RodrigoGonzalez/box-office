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

f = codecs.open("index.html?page=daily&view=chart&id=marvel2016.htm&adjust_yr=2016&p=.htm", 'r',"utf-8")
def get_revenue_table(file):
    soup = BeautifulSoup(f, "html5lib")
    java_text = soup.find_all(type="text/javascript")
    t = java_text[5]
    jtext = t.getText().split('\t')[6].replace("\n", " ")
    js2py.eval_js(jtext)
    revenue = table.to_list()
    title = soup.title.get_text().replace(" - Daily Box Office Results - Box Office Mojo", "")
    r = soup.body.find(id="container").find(id="main").find(id="body").select("table")[2].tbody.tr.td.select("table")[0].tbody.tr.center.tbody.b
    total_revenues = re.sub('[!@#$,]', '', r.get_text())

    return unidecode(title), int(total_revenues)


    from urllib2 import Request, urlopen, URLError


    # To strip html from critic reviews

    url = 'http://christylemire.com/captain-america-civil-war/'
    page = urllib2.urlopen(url)
