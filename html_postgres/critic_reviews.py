import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
from unidecode import unidecode
import re
import requests
import bleach

def load_dataframe():
    df = pd.read_csv('movie_revs.csv')
    df['movie_key'] = ["".join([re.sub('[!@#$/:,]', '', x) for x in y.lower().split()]) for y in df['title2']]
    df['match_key'] = df['title1'].apply(keyify)
    return df

def keyify(string):
    return set(re.sub('\W', ' ', string).lower().split())
    # for flagging

def extract_text(link):
    r = requests.get(link)

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

    return perfect

def 

if __name__ == '__main__':
    df = load_dataframe()
    columns = ['id', 'movie_key', 'critic', 'review']
    df_reviews = pd.DataFrame(columns=columns)

Use
if r.status_code == requests.codes.ok:
r.status_code
