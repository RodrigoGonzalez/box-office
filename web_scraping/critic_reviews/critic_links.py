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

def get_all_titles():
    """ returns all the movie names from boxofficemojo.com in a list"""
    # The following imports the list of box office mojo URLs
    list_urls = pd.read_csv('../get_movie_information/summary.txt', header = None, sep='\t')

    # Set index to urls
    index = list_urls.values.T.tolist()

    # List of movie urls
    titles_list = []
    title_year = []
    movie_year = []
    dead_links = []

    # Loop through the pages for each letter
    for i, link in enumerate(index[0][:]):
        try:
            page = urllib2.urlopen(link)
            soup = BeautifulSoup(page, "html5lib")
            # Removing "- Box Office Mojo" from the title names
            title = soup.find("title").contents[0].strip('- Box Office Mojo')

                # print i
                # print title[:-6]
                # print title[-6:].strip("()")

            # Split title into name and year
            titles_list.append(title[:-6])
            title_year.append(title[:-6] + ", " + title[-6:].strip("()"))
            movie_year.append(title[-6:].strip("()"))
            time.sleep(0.2)

        except Exception, e:
            print link
            dead_links.append(link)
            logging.exception(e)

    return titles_list, title_year, movie_year

# def critic_review_links():
#

if __name__ == '__main__':
    titles_list, title_year, movie_year = get_all_titles()

# seperate timeouts on aws, compartmentalize code
