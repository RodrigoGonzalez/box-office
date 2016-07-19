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
            title_year.append(title)
            movie_year.append(title[-6:].strip("()"))
            time.sleep(0.15)

        except Exception, e:
            print link
            dead_links.append(link)
            logging.exception(e)

    titles_list = [unidecode(title) for title in titles_list]
    title_year = [unidecode(title_year) for title_year in title_year]
    movie_year = [unidecode(title) for movie_year in movie_year]

    return titles_list, title_year, movie_year, dead_links

# def critic_review_links():
#

if __name__ == '__main__':
    titles_list, title_year, movie_year, dead_links = get_all_titles()

    # Save lists to csv files
    np.savetxt("titles_list.csv", titles_list, delimiter=",", fmt='%s')
    np.savetxt("title_year.csv", title_year, delimiter=",", fmt='%s')
    np.savetxt("movie_year.csv", movie_year, delimiter=",", fmt='%s')
    np.savetxt("dead_links.csv", dead_links, delimiter=",", fmt='%s')


# seperate timeouts on aws, compartmentalize code, distributed scraping
# np.savetxt("titles_list.csv", test, delimiter=",", fmt='%s')
