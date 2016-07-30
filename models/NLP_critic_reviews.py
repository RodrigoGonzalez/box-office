import numpy as np
import pandas as pd
import unidecode as unidecode
import re, os
from string import punctuation
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import dill as pickle
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer



def load_critic_reviews():
    
    return df

def main():
	df_basedata = pd.read_csv('../html_postgres/movie_revs.csv')
    pass


if __name__ == '__main__':
    main()
