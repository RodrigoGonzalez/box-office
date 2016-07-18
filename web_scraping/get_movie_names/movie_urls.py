import urllib2
from bs4 import BeautifulSoup
import logging
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
import string
import cPickle as pickle

def get_all_movies():
    """ returns all the movie urls from boxofficemojo.com in a list"""

    # Alphabet loop for how movies are indexed including
    # movies that start with a special character or number
    index = ["NUM"] + list(string.ascii_uppercase)

    # List of movie urls
    movies_list = []

    # Loop through the pages for each letter
    for letter in index:

        # Loop through the pages within each letter
        for num in range(1, 20):
            url = ("http://www.boxofficemojo.com/movies/alphabetical.htm?"
                   "letter=" + letter + "&page=" + str(num))
            try:
                page = urllib2.urlopen(url)
                soup = BeautifulSoup(page, "html5lib")
                rows = soup.find(id="body").find("table").find("table").find_all(
                    "table")[1].find_all("tr")

                # skip index row
                if len(rows) > 1:
                    counter = 1
                    for row in rows:

                        # skip index row
                        if counter > 1:
                            link = row.td.font.a['href']

                            # don't add duplicates
                            if link not in movies_list:
                                movies_list.append(link)

                        counter += 1
            except Exception, e:
                logging.exception(e)

    return movies_list

if __name__ == '__main__':
    movie_names = get_all_movies()
    movie_names[4427] = '/movies/?id=elizabeth%A0.htm'
    movie_names[12973] = '/movies/?id=simpleplan%A0.htm'
    movies = pd.DataFrame(movie_names)
    movies['movie'] = [x[1] for x in movies[0].str.split('?')]
    movies['movie'].to_csv('movie_names.csv',index=False)

    # with open('movie_names.pkl', 'w') as f:
    #     pickle.dump(movie_names, f)
    # with open('movie_names.pkl') as f:
    #     model = pickle.load(f)


    # Problems
    # id=elizabeth%A0.htm - correct
    # id=elizabeth\xa0.htm - html5lib parser
    # id=elizabeth .htm - when read into DataFrame
