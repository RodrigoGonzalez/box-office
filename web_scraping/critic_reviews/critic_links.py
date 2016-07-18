import pandas as pd
import numpy as np
import urllib2
from bs4 import BeautifulSoup
import logging
logging.basicConfig(level=logging.DEBUG)
import string
import cPickle as pickle



def get_all_titles(list):
    """ returns all the movie names from boxofficemojo.com in a list"""
    # The following imports the list of box office mojo URLs
    list_urls = pd.read_csv('../get_movie_information/summary.txt', header = None, dtype="string")
    list_urls = np.loadtxt('../get_movie_information/summary.txt', dtype="string")

    i = soup.find("title")


if __name__ == '__main__':
    main()
