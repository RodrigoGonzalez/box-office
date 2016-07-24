from selenium import webdriver
import pandas as pd
import numpy as np
import urllib2
from bs4 import BeautifulSoup
# import logging
# logging.basicConfig(level=logging.DEBUG)
import string
import cPickle as pickle
from fake_useragent import UserAgent
from pyvirtualdisplay import Display
from unidecode import unidecode


# Use PhantomJS and set window
# driver = webdriver.PhantomJS()
# driver.set_window_size(1120, 550)
# Run initially to keep display from opening up

display = Display(visible=0, size=(1920, 1080))
display.start()

# Google Chrome to scrape (on aws linux chromedriver at
path_to_chromedriver = '/usr/local/bin/chromedriver'

# Set option for chromebrowser
chrome_options = webdriver.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images":2}
chrome_options.add_experimental_option("prefs",prefs)
chrome_options.add_argument("user-agent={}".format(UserAgent().random))


# change path as needed to where web driver lives
browser = webdriver.Chrome(executable_path = path_to_chromedriver, chrome_options=chrome_options)
