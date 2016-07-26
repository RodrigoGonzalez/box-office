import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def process_genres(string):
    """
    INPUT:
    OUTPUT:
    """
    return "/".join(string.replace("/", "").split())

def process_distributor(string):
    """
    INPUT:
    OUTPUT:
    """
    return "/".join([x.strip() for x in string.split("/")])

def load_dataframe():
    """
    INPUT: filename preprocessed revenue data
    OUTPUT: dataframe ready for modelling
    """

    df = pd.read_csv('../html_postgres/movie_revs.csv')
    df['distributor'] = df['distributor'].apply(process_distributor)
    df['genre'] = df['genre'].apply(process_genres)

    distributors = df.pop('distributor').str.get_dummies(sep="/")
    genres = df.pop('genre').str.get_dummies(sep="/")




def
if __name__ == '__main__':
    main()
