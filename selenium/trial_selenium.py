from selenium import webdriver
import pandas as pd
import numpy as np
import urllib2
from bs4 import BeautifulSoup
import logging
logging.basicConfig(level=logging.DEBUG)
import string
import cPickle as pickle


# Google Chrome to scrape
path_to_chromedriver = '/Users/rodrigogonzalez/Dropbox/Galvanize/box-office/selenium/chromedriver'

# change path as needed to where web driver lives
browser = webdriver.Chrome(executable_path = path_to_chromedriver)

# Go to page
url = 'https://www.rottentomatoes.com/'
browser.get(url)

# For rottentomatoes
# name = 'search'
# id = 'search-term'

# Clear the search field, and input the search term
browser.find_element_by_id('search-term').clear()
browser.find_element_by_id('search-term').send_keys('Gladiator')

# Click search
browser.find_element_by_xpath('//*[@id="header_brand_column"]/div[1]/form/div/div/div[1]/button').click()

# First Link
# //*[@id="movie_results_ul"]/li[1]/div/div/a
# or //*[@id="movie_results_ul"]/li/div/div/a

# Figure out how to use contains term
# dyn_frame = browser.find_element_by_xpath('//*[contains(@id, "movie_results_ul")]')

# Find the first search result (default)


dyn_frame = browser.find_element_by_xpath('//*[@id="movie_results_ul"]/li/div/div/a')

linkname = dyn_frame.get_attribute('href')

# Go to the movie URL
browser.get(linkname)
dyn_frame2 = browser.find_element_by_xpath('//*[@id="criticHeaders"]/a[2]').click()

# Get all reviews links in id = "content"

dyn_frame3 = browser.find_elements_by_xpath('//*[contains(@rel, "nofollow")]')


for link in dyn_frame3:
    print link.get_attribute('href')

//*[@id="reviews"]/div[2]/div[2]/div[1]/div[2]/div[2]/div[2]/div[2]/a
//*[@id="reviews"]/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/a
