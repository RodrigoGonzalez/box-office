import os, re
import pandas as pd
import numpy as np
from lxml import html
import cPickle as pickle

from passage.utils import save, load
from model_RNN_Adadelta import load_reviews

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
        - stripped and clean title2
    Returns cleaned movie_key
    """
    return set(re.sub('\W', ' ', string).lower().split())
    # for flagging

def generate_word_set():
    """
    INPUT: list of strings
        - Contains the reviews to be cleaned
    Returns a set including all of the words present in the training data
    """
    data_path = '../data/reviews_sentiment/'
    file_name = f'{data_path}labeledTrainData.tsv'

    # Load Data
    X, y = load_reviews(file_name)

    included_words = set()

    for review in X:
        included_words.update(set(review.split(" ")))

    return included_words

def clean(texts, ws):
    """
    INPUT: list of strings (html)
    Returns text content (and the text in any children), stripped, lowercase from a single document.
    """
    revs = [html.fromstring(text).text_content().lower().strip() for text in texts]

    cln_reviews = []

    for r in revs:
        words = r.split(" ")
        include = [word for word in words if word in ws and word]
        if not include:
            include.append('empty')
        cln_reviews.append(" ".join(include))
    return cln_reviews

def process_review_data(df, movie_key, ws):
    """
    INPUT: Pandas DataFrame
        - A data frame containing the critic reviews to be processed
    Returns three lists corresponding to reviews, labels for each review, and the movie_keys
    """
    m_key = re.sub('.csv', '', movie_key)

    df_review = df[df['status_code'] == 200].copy()
    print df_review.shape

    if df_review.shape[0] == 0:
        print movie_key
        return '0', m_key, m_key
    else:
        try:
            X_ = clean(df_review['review'].values, ws)
            j_reviews = [" ".join(X_)]
            if type(j_reviews) == list:
                j_reviews = j_reviews[0]
            X_.append(j_reviews)

        except (RuntimeError, TypeError, NameError):
            print m_key

        Id = df_review['id'].tolist()
        Id.append('{}_tot'.format(m_key))
        movie_keys = [m_key for x in xrange(len(Id))]

        return X_, Id, movie_keys

def movie_reviews(path):
    """
    INPUT: string
        - The path of folder containing the movie reviews for each movie
    Returns a dataframe with all of the processed movie reviews
    """
    reviews_loc = filter(lambda x: 'csv' in x, os.listdir(path))

    X, IDs, keys = [], [], []

    ws = generate_word_set()

    print "\nBegin loading moview reviews"

    for j, review in enumerate(reviews_loc):
        df = pd.read_csv(path+review)
        df.dropna(axis=0, inplace=True)
        x, i, k = process_review_data(df, review, ws)

        op = "append" if len(x) == 1 else "extend"

        eval("X.{0}(x)".format(op))
        eval("IDs.{0}(i)".format(op))
        eval("keys.{0}(k)".format(op))

        if len(X) != len(keys):
            print len(X), len(keys)
            print j, review
            raise ValueError('Arrays no longer match')

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
    df  = df[df.reviews != '0'].copy()
    X  = df.pop('reviews').values

    # Load models and text tokenizer
    model, tokenizer = sentiment_scorer()

    print "\nTokenizing reviews"

    # Tokenize reviews
    X_tokens = tokenizer.transform(X)

    print "\nMaking sentiment predictions"
    # Make sentiment predictions
    y_pred = model.predict(X_tokens).flatten()
    df['sentiment'] = y_pred.reshape(-1, 1)

    # Save to csv file sentiment predictions
    file_save = '../data/reviews_sentiment/sentiment1.csv'
    df.to_csv(file_save, header=True, index=False)

# movie_revs movie revenues 0 - 1957
# search titles 0 - 1957
# critic links2 0 - 1943
# critic_reviews 0 - 9144
