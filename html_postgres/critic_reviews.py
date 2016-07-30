from __future__ import division
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
from unidecode import unidecode
import re, os
import requests
import bleach
import logging
logging.basicConfig(level=logging.DEBUG)
import math
import multiprocessing
import itertools
from timeit import Timer

def load_dataframe():
    df = pd.read_csv('movie_revs.csv')
    df['movie_key'] = ["".join([re.sub('[!@#$/:,]', '', x) for x in y.lower().split()]) for y in df['title2']]
    df['match_key'] = df['title1'].apply(word_set)
    return df

def word_set(string):
    return set(re.sub('\W', ' ', string).lower().split())
    # for flagging

def include_review(title, main_text):
    x = np.sum([1 for word in title if word in main_text])
    return x

def extract_text(link):
    try:
        r = requests.get(link)
        stat_code = r.status_code
        f = r.text

        soup = BeautifulSoup(f, "html5lib")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        txt = soup.get_text()

        # bleach out the remaining links
        bleached = bleach.clean(txt, strip=True)

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in bleached.splitlines())

        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # clean
        white = re.sub('[!@#$:,&*]', '', text)

        # clean some more
        snow_white = unidecode(re.sub('\s', ' ', white))

        # perfect
        perfect = re.sub('[*\']', '', snow_white)

        return perfect, stat_code

    except Exception, e:
        missing_information.append(link)
        logging.exception(e)
        perfect = '0'
        status_code = '0'
        return perfect, status_code

def list_of_links(file):
    data = []
    stat_codes = []
    with open(file) as f:
        links = [re.sub('[\n]', "", link) for link in f if link != '0']
    links.pop(0)

    pool = multiprocessing.Pool(4)  # for each core
    output = pool.map(extract_text, links)
    df_reviews = pd.DataFrame(output, columns=['data', 'stat_codes'])
    review_data = zip(df_reviews['data'], links, df_reviews['stat_codes'])
    return review_data

if __name__ == '__main__':
    missing_information = []
    done = []
    df = load_dataframe()
    columns = ['id', 'movie_key', 'critic', 'link', 'review', 'status_code']
    df_reviews = pd.DataFrame(columns=columns)

    path = '../critic_links2'
    files = os.listdir(path)

    for filename in files:
        file = path + "/" + filename
        columns = ['review', 'link', 'status_code']
        review_data = pd.DataFrame(list_of_links(file), columns=columns)
        id_base = re.sub('.txt', '', filename)
        review_data['id'] = [id_base + str(item) for item in xrange(review_data.shape[0])]
        review_data['movie_key'] = re.sub('.txt', '', filename)
        review_data['critic'] = [link.split('.')[1]  for link in review_data['link']]
        review_data['status_code'] = review_data['status_code'].astype(int)
        review_data.to_csv('../critic_reviews/{}.csv'.format(re.sub('.txt', '', filename)), mode = 'w', index = False)
        df_reviews = pd.concat([df_reviews, review_data], axis=0)
        df_reviews = df_reviews.append(review_data, ignore_index=True)
        done.append(filename)
        print filename

    df_reviews.to_csv('critic_reviews.csv', mode = 'w', index=False)
    movies_done = pd.DataFrame(done)
    movies_done.to_csv('movies_done.csv', mode = 'w', header=False, index=False)

    missing_info = pd.DataFrame(missing_information)
    missing_info.to_csv('missing_info.csv', mode = 'w', header=False, index=False)
