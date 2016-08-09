import pandas as pd
import numpy as np
import urllib2
from bs4 import BeautifulSoup
import logging
logging.basicConfig(level=logging.DEBUG)
import string
import cPickle as pickle
import re
import time
import csv
from unidecode import unidecode
from selenium import webdriver

path_to_chromedriver = '/Users/rodrigogonzalez/Dropbox/Galvanize/box-office/selenium/chromedriver'
# change path as needed

browser = webdriver.Chrome(executable_path = path_to_chromedriver)

url = 'https://www.wikipedia.org/'
browser.get(url)

# For rottentomatoes
# name = 'search'
# id = 'search-term'

# Search the main wikipedia page
browser.find_element_by_id('searchInput').clear()
browser.find_element_by_id('searchInput').send_keys('Gladiator (2000)')

# Click on the search result
browser.find_element_by_xpath('//*[@id="search-form"]/fieldset/button').click()

dyn_frame = browser.find_element_by_xpath('//*[@id="mw-content-text"]/div/ul/li[1]/div[1]/a')

framename = dyn_frame.get_attribute('href')

browser.find_element_by_id('Reception')
# Go to the movie URL
browser.get('view-source:' + framename)

url = (framename)
try:
    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page, "html5lib")

    rows = soup.find(id="mw-content-text")
