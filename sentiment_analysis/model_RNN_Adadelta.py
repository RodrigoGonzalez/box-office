import numpy as np
import pandas as pd
from lxml import html

from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.preprocessing import Tokenizer
from passage.utils import save, load

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score)

def load_reviews(file_name):
	"""
	INPUT: string
		- filename containting the reviews data
	Return array(s) corresponding to positive or negative reviews
	"""
	df = pd.read_csv(file_name, delimiter='\t')
	X = clean(df['review'].values)

	print "Reviews loaded and cleaned"

	if 'sentiment' in df:
		y = df['sentiment'].values
		return (X, y)
	else:
		return X

def clean(texts):
	"""
	INPUT: list of strings (html)
	Returns text content (and the text in any children), stripped, lowercase from a single document.
	"""
	return [html.fromstring(text).text_content().lower().strip() for text in texts]

def tokenize(train):
	"""
	INPUT: Array
		- Text documents (reviews) to train sentiment on
	Returns trained tokenizer
	"""
	tokenizer = Tokenizer(min_df=10, max_features=100000)
	print "Training tokenizer on reviews"
	tokenizer.fit(train)
	return tokenizer

def train_RNN(tokenizer, tokens, labels):
	"""
	INPUT: Trained tokenizer class, label array
		- The arrays of the tokenized critic reviews and the corresponding labels
	Returns a trained Recurrent Neural Network class object
	"""
	layers = [
		Embedding(size=256, n_features=tokenizer.n_features),
		GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', seq_output=False, p_drop=0.75),
		Dense(size=1, activation='sigmoid', init='orthogonal')
	]

	model = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))

	path_snapshots = 'model_snapshots'

	print "Begin fitting RNN"

	model.fit(tokens, labels, n_epochs=10, path=path_snapshots)

	return model

if __name__ == "__main__":

	data_path = '../data/reviews_sentiment/'
	file_name = data_path + 'labeledTrainData.tsv'

	# Load Data
	X, y = load_reviews(file_name)

	# Split to training and testing data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=74)

	# Train tokenizer
	tokenizer = tokenize(X_train)
	X_tokens = tokenizer.transform(X_train)

	# Train Recurrent Neural Network
	model = train_RNN(tokenizer, X_tokens, y_train)

	y_pred_tr = model.predict(X_tokens).flatten()

	# Check overall performance
	test_tokens = tokenizer.transform(X_test)
	y_pred_tst = model.predict(test_tokens).flatten()

	scorers = [roc_auc_score, precision_score, recall_score, f1_score]

	for score in scorers:
		tr_score = score(y_train, y_pred_tr)
		t_score = score(y_test, y_pred_tst)
		print "Scoring method: {0}".format(score.func_name.title())
		print "Train: {0:.2f}, Test: {1:.2f}".format(tr_score, t_score)
		# '{0:.{1}f}'.format(f, 2)

	# Save model for future use
    save(model, 'review_scorer.pkl')
    # model = load('review_scorer.pkl')
