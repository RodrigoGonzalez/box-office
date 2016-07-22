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

f = codecs.open("index.html?page=daily&view=chart&id=marvel2016.htm&adjust_yr=2016&p=.htm", 'r',"utf-8")

soup = BeautifulSoup(f, "html5lib")
tag = soup.script

print tag.keys()

# soup.body.find(id="container").find(id="main").find(id="body").contents[-2].tbody.tr.td
