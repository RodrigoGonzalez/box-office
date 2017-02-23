# box-office

## Rodrigo Gonzalez's Capstone Project at Galvanize, Inc.

### Overview (Currently cleaning python code 02.15.2017)

box-office predicts movie box office revenues of feature length films to identify stock market opportunities in media properties. The tool is based on critic reviews, film characteristics, production budget, and what studio and players are involved. Producing a movie is a highly risky endeavor and studios rely on only a handful of extremely expensive movies every year to make sure they remain profitable. Box office hits and misses correspond to short-term changes in stock prices of media properties.

Project utilizes web scraping, (Natural Language Processing, [NLP](en.wikipedia.org/wiki)) on [SENTIMENT ANALYSIS](https://en.wikipedia.org/wiki/Sentiment_analysis), and feature selection to identify factors that best predict box office success using machine learning techniques (Ensemble Methods including Random Forest & Boosting, along with a Recurrent Neural Network for sentiment analysis and Clustering Methods for binning individual features) and big data analytics.

### Motivation

box-office is a revenue predicting tool for feature length films. The movie industry is a multi-billion dollar industry, generating approximately $40 billion of revenue annually worldwide. However, investing in the production of a feature length film is a highly risky endeavor and studios rely on only a handful of extremely expensive movies every year to make sure they remain profitable. Over the last decade, 80% of the industry’s profits was generated from just 6% of the films released; and 78% of movies have lost money of the same time period.

According to Jack Valenti, President and CEO of the Motion Picture Association of America (MPAA):
“No one can tell you how a movie is going to do in the marketplace. Not until the film opens in darkened theatre and sparks fly up between the screen and the audience.”

This project aims to identify the predictive features of box office revenues, which will help studios and investors better measure the risk taken on producing different films, helping the stake-holders to better plan for execute movies that audiences will enjoy and are financially profitable.

The aim of box-office is to:
* Predict total revenue of feature length films by investigating extent to which players involves and movie characteristics determine the overall market success of the film
* Examine the impact of positive or negative critic reviews
* Examine the relationship between weekend box-office revenues and the stock prices of the media and entertainment companies involved


### Data

#### Data Sources:

* Rotten Tomatoes (critic reviews): [Rotten Tomatoes](https://www.rottentomatoes.com): Film review aggregator, a site where people can get access to reviews from a variety of critics in the United States.
* General Movie Information (ID, title, year, actors, producer, directors, writers, lifetime earnings): [IMDb](imdb.com): An online database of information related to films & television programs, including cast, production crew, fictional characters, biographies, plot summaries, trivia and reviews.
* Federal Reserve Economic Data (Macroeconomic Indicators) [FRED](https://fred.stlouisfed.org): A database maintained by the Research division of the Federal Reserve Bank of St. Louis.
* BoxOffice Mojo (daily box office revenues, production budget, premier date, genre, production company/studio): [BoxOffice Mojo](http://www.boxofficemojo.com): Tracks box office revenue in a systematic, algorithmic way
* Yahoo Finance API (S&P500 daily index, media company stock prices): [Yahoo Finance API](http://finance.yahoo.com): Yahoo finance provides stock data.
* IMDB movie reviews[](https://www.kaggle.com/c/word2vec-nlp-tutorial/data):

#### Data Scope:

The data set used in this project consists of over 75,000 critic reviews from a hundred or so publications, and the full archives of the Internet Movie Database, which was loaded into 43 different tables, some of which had more than 50 million individual data entries. A number of macroeconomic indicators were used as well.


### Analysis:

Analysis began with the collection of movie data from boxoffice mojo.

### Tools

1. [Python](https://www.python.org/): the main coding language for this project.
2. [Beautiful Soup](http://www.crummy.com/software/Beautifulsoup/): a Python library designed for web-scraping. It provides strong parse power especially HTML.
5. [NLTK](http://www.nltk.org/): Natual Language Toolkit, a Python library that provides support for Natural Language Processing including stopwords lists, word Stemmer and Lemmatizer and etc.
6. [sklearn](http://scikit-learn.org/): Scikit-Learn, a Python library that provides all sorts of machine learning libraries and packages.
7. [Flask](http://flask.pocoo.org/): a microframework for Python based on Werkzeug, Jinja 2.
8. [d3.js](http://d3js.org/): Data-Driven Documents, a JavaScript Library that helps interactively visualizing data and telling stories about the data.
9. [nvd3](http://nvd3.org/): a JavaScript wrapper for d3.js.
10. [word2vec](https://en.wikipedia.org/wiki/Word2vec): used for learning vector representations of words, called "word embeddings". These representations can be subsequently used in many natural language processing applications and for further research.


### Credits and Acknowledge

A special thank you to:

* [rottentomatoes.com](https://www.rottentomatoes.com) for providing the critic reviews
* [imdb.com](imdb.com) for providing the majority of movie data
* Fellow Students and Instructors at [Galvanize gSchool / Zipfian Academy](http://www.zipfianacademy.com/) for providing the tools and background necessary to complete this project.
