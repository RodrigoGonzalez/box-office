from selenium import webdriver
import pandas as pd
import numpy as np
import urllib2
from bs4 import BeautifulSoup
import logging
logging.basicConfig(level=logging.DEBUG)
import string
import cPickle as pickle
from fake_useragent import UserAgent
from pyvirtualdisplay import Display
from unidecode import unidecode
from xvfbwrapper import Xvfb
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.keys import Keys
import sys, getopt
import re

# vdisplay = Xvfb
# vdisplay.start()

# Use PhantomJS and set window
# driver = webdriver.PhantomJS()
# driver.set_window_size(1120, 550)
# Run initially to keep display from opening up
def extract_links(row):
    display = Display(visible=0, size=(1920, 1080))
    display.start()

    path_to_chromedriver = '/Users/rodrigogonzalez/Desktop/chromedrive3/chromedriver'

    # Set option for chromebrowser
    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.managed_default_content_settings.images":2}
    chrome_options.add_experimental_option("prefs",prefs)
    chrome_options.add_argument("user-agent={}".format(UserAgent().random))
    chrome_options.add_argument("--disable-bundled-ppapi-flash");


    # change path as needed to where web driver lives
    browser = webdriver.Chrome(executable_path = path_to_chromedriver, chrome_options=chrome_options)

    # Go to page
    url = row['link']
    movie_name = row['title']
    browser.get(url)

    try:
        # # Clear the search field, and input the search term
        # browser.find_element_by_id('search-term').clear()
        # browser.find_element_by_id('search-term').send_keys(re.sub(':', ' : ', movie_name))
        #
        # # Click search
        # browser.find_element_by_xpath('//*[@id="search-form"]/div/div/div[1]/div[1]/button/em').click()

        # # Find the first search result (default)
        # dyn_frame = browser.find_element_by_xpath('//*[@id="movie_results_ul"]/li/div/div/a')
        #
        # linkname = dyn_frame.get_attribute('href')
        #
        # # Go to the movie URL
        # browser.get(linkname)
        # dyn_frame2 = browser.find_element_by_xpath('//*[@id="criticHeaders"]/a[2]').click()

        # Get all reviews links in id = "content"

        dyn_frame3 = browser.find_elements_by_xpath('//*[contains(@rel, "nofollow")]')

        links = []

        for link in dyn_frame3:
            links.append(unidecode(link.get_attribute('href')))
            print unidecode(link.get_attribute('href'))

        # key
        filename = "".join([re.sub('[!@#$/:,]', '', x) for x in movie_name.lower().split()])
        pd.DataFrame(links).to_csv('../critic_links2/'+ filename +'.txt', sep='\t', index=False)

    except Exception, e:
        empty_keys.append(movie_name)
        logging.exception(e)# look at critic names later

    browser.quit() #closes browser

if __name__ == '__main__':
    empty_keys = []
    df = pd.read_csv('fourth_pass.csv')

    columns = ['title', 'link']

    for index, row in df.iterrows():
        extract_links(row)
        # empty_keys.append(title)
        print empty_keys
    np.savetxt("pass4_empty_keys" + ".csv", empty_keys, delimiter=",", fmt='%s')
