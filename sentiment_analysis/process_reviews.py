import os, re
import pandas as pd
import numpy as np
from lxml import html
import cPickle as pickle

from passage.utils import save, load

def load_dataframe():
    """

    """
    df = pd.read_csv('../html_postgres/movie_revs.csv')
    df['movie_key'] = ["".join([re.sub('[!@#$/:,]', '', x) for x in y.lower().split()]) for y in df['title2']]
    df['match_key'] = df['title1'].apply(word_set)
    return df

def word_set(string):
    """
    INPUT: string
        -
    Returns
    """
    return set(re.sub('\W', ' ', string).lower().split())
    # for flagging

def clean(texts):
	"""
	INPUT: list of strings (html)
	Returns text content (and the text in any children), stripped, lowercase from a single document.
	"""
	return [html.fromstring(text).text_content().lower().strip() for text in texts]

def process_review_data(df, movie_key):
    """
    INPUT: Pandas DataFrame
        - A data frame containing the critic reviews to be processed
    Returns three lists corresponding to reviews, labels for each review, and the movie_keys
    """
    m_key = re.sub('.csv', '', movie_key)

    df_review = df[df['status_code'] == 200].copy()
    try:
        X = clean(df_review['review'].values)
        X.append(" ".join(X))

    except (RuntimeError, TypeError, NameError):
        print m_key

    Id = df_review['id'].tolist()
    Id.append('{}_tot'.format(m_key))
    movie_keys = [m_key for x in xrange(len(Id))]

    return X, Id, movie_keys

def movie_reviews(path):
    """
    INPUT: string
        - The path of folder containing the movie reviews for each movie
    Returns a dataframe with all of the processed movie reviews
    """
    reviews = filter(lambda x: 'csv' in x, os.listdir(path))

    X, IDs, keys = [], [], []

    print "\nBegin loading moview reviews"

    for review in reviews:
        df = pd.read_csv(path+review)
        df.dropna(axis=0, inplace=True)
        x, i, k = process_review_data(df, review)
        X.extend(x)
        IDs.extend(i)
        keys.extend(k)

    df = pd.DataFrame({ 'reviews'  : X,
                        'review_id': IDs,
                        'movie_key': keys
                        })

    return df

def sentiment_scorer():
    """
    INPUT: None
    Returns the previously trained RNN sentiment scoring model and the tokenizer used
    """
    print "\nLoading tokenizer"

    with open('review_tokenizer.pkl','r') as fileObject:
        tokenizer = pickle.load(fileObject)

    print "\nLoading Recurrent Neral Network model"

    model = load('review_scorer.pkl')

    print "\nDone loading models"

    return model, tokenizer

if __name__ == '__main__':

    # Movie reviews path
    path = '../critic_reviews/'

    # Load reviews
    df = movie_reviews(path)
    X  = df.pop('reviews').values

    # Load models and text tokenizer
    model, tokenizer = sentiment_scorer()

    print "\nTokenizing reviews"

    # Tokenize reviews
    X_tokens = tokenizer.transform(X)

    print "\n Making sentiment predictions"
    # Make sentiment predictions
    y_pred = model.predict(X_tokens).flatten()
    df['sentiment'] = y_pred.reshape(-1, 1)

    # Save to csv file sentiment predictions
    file_save = '../data/reviews_sentiment/sentiment.csv'
    df.to_csv(file_save, header=True, index=False)

# movie_revs movie revenues 0 - 1957
# search titles 0 - 1957
# critic links2 0 - 1943
# critic_reviews 0 - 9144
