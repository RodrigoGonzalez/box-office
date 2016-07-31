import numpy as np
import pandas as pd
from lxml import html
import multiprocessing
import unidecode as unidecode
import re, os
from timeit import Timer

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, LstmRecurrent, Dense
from passage.preprocessing import Tokenizer
import dill as pickle



filename =  'madmaxfuryroad.csv'

def critic_name(string):
    rep = {"http://": "", "www.": "", ".html": "", ".php": ""}
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.iteritems())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], string)
    units = [x for x in text.split(".") if "/" not in x]
    return "_".join([i for i in units if not i.isdigit()])

def clean_reviews(filename):
    columns = ['review', 'link', 'status_code', 'id', 'movie_key', 'critic']
    dir_path = '../critic_reviews'
    dfm = pd.read_csv(dir_path + "/" + str(filename))
    dfm = dfm[dfm.status_code == 200]
    dfm['critic'] = dfm['link'].apply(critic_name)
    print "Cleaned: {}".format(filename)
    return dfm

def load_data():
    columns = ['review', 'link', 'status_code', 'id', 'movie_key', 'critic']
    df = pd.DataFrame(columns=columns)
    dir_path = '../critic_reviews'
    filenames = os.listdir(dir_path)
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)
    output = pool.map(clean_reviews, filenames)

    print "Merging all frames"
    i = 0
    for frame in output:
        i += 1
        df = df.append(frame ,ignore_index=True)
        if i % 100 == 0:
            print "Now on {}th iteration".format(i)
    print "Finished loading data"
    return df

def clean(texts):
	return [html.fromstring(text).text_content().lower().strip() for text in texts]

def train_model(modeltype):

    assert modeltype in ["gated_recurrent", "lstm_recurrent"]
    print "Begin Training"

    df_imdb_reviews = pd.read_csv('../data/imdb_review_data.tsv', escapechar='\\', delimiter='\t')

    X = clean(df_imdb_reviews['review'].values)
    y = df_imdb_reviews['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print "Tokenize"

    tokenizer = Tokenizer(min_df=10, max_features=100000)
    X_train = tokenizer.fit_transform(X_train)

    print "Number of featers: {}".format(tokenizer.n_features)

    print "Training model"

    if modeltype == "gated_recurrent":
        layers = [
            Embedding(size=256, n_features=tokenizer.n_features),
            GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                           init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]
    else:
        layers = [
            Embedding(size=256, n_features=tokenizer.n_features),
            LstmRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                          init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]

    # bce is classification loss for binary classification and sigmoid output
    model = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))
    model.fit(X_train, y_train, n_epochs=10)

    with open('../data/{}_tokenizer.pkl'.format(modeltype), 'w') as f:
        vectorizer = pickle.dump(tokenizer, f)
    with open('../data/{}_model.pkl'.format(modeltype), 'w') as f:
        model = pickle.dump(model, f)

    y_pred_te = model.predict(y_test).flatten() >= 0.5
    y_pred_tr = model.predict(ytrain).flatten() >= 0.5
    try:
        print 'Test Accuracy: {}'.format(accuracy_score(y_test,y_pred_te))
        print 'Test Precision: {}'.format(precision_score(y_test,y_pred_te))
        print 'Test Recall: {}'.format(recall_score(y_test,y_pred_te))
        print 'Train Accuracy: {}'.format(accuracy_score(y_train,y_pred_tr))
        print 'Train Precision: {}'.format(precision_score(y_train,y_pred_tr))
        print 'Train Recall: {}'.format(recall_score(y_train,y_pred_tr))

    except:
        print "Unable to perform metrics"

    return tokenizer, model


def main():
    df = load_data()
    tokenizer1, model1 = train_model("gated_recurrent")
    tokenizer2, model2 = train_model("lstm_recurrent")

    X_rev = clean(df['review'].values)
    X_rev_gated = tokenizer1.transform(X_rev)
    df['probabilities_gated'] = model.predict(X_rev_gated).flatten()
    X_rev_lstm = tokenizer2.transform(X_rev)
    df['probabilities_lstm'] = model.predict(X_reX_rev_lstmv1).flatten()

    return df


if __name__ == "__main__":
    df = main()
    df.to_csv('../data/review_predictions.csv', mode = 'w', index=False)


# Load below
# with open('../data/{}_vectorizer.pkl'.format(modeltype)) as f:
#     vectorizer = pickle.load(f)
# with open('../data/{}_model.pkl'.format(modeltype)) as f:
#     model = pickle.load(f)
