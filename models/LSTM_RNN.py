from __future__ import print_function
import time
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, BatchNormalization
from keras.datasets import imdb

import os
from keras.preprocessing.text import Tokenizer

max_features = 10000
max_len = 200  # cut texts after this number of words (among top max_features most common words)

X_train = []
y_train = []

path = './aclImdb/train/pos/'
X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
y_train.extend([1 for _ in range(12500)])

path = './aclImdb/train/neg/'
X_train.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
y_train.extend([0 for _ in range(12500)])

print('x:')
print(X_train[:1])
print(X_train[-1:])
print(len(X_train))
print('y:')
print(y_train[:1])
print(y_train[-1:])
print(len(y_train))

# read in the test data

X_test = []
y_test = []

path = './aclImdb/test/pos/'
X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
y_test.extend([1 for _ in range(12500)])

path = './aclImdb/test/neg/'
X_test.extend([open(path + f).read() for f in os.listdir(path) if f.endswith('.txt')])
y_test.extend([0 for _ in range(12500)])

print('x:')
print(X_test[:1])
print(X_test[-1:])
print(len(X_test))
print('y:')
print(y_test[:1])
print(y_test[-1:])
print(len(y_test))

#tokenize works to list of integers where each integer is a key to a word
imdbTokenizer = Tokenizer(nb_words=max_features)

imdbTokenizer.fit_on_texts(X_train)

#print top 20 words
#note zero is reserved for non frequent words
for word, value in imdbTokenizer.word_index.items():
    if value < 20:
        print(value, word)

#create int to word dictionary
intToWord = {}
for word, value in imdbTokenizer.word_index.items():
    intToWord[value] = word

#add a symbol for null placeholder
intToWord[0] = "!!!NA!!!"

print(intToWord[1])
print(intToWord[2])
print(intToWord[32])

#convert word strings to integer sequence lists
print(X_train[0])
print(imdbTokenizer.texts_to_sequences(X_train[:1]))
for value in imdbTokenizer.texts_to_sequences(X_train[:1])[0]:
    print(intToWord[value])

X_train = imdbTokenizer.texts_to_sequences(X_train)
X_test = imdbTokenizer.texts_to_sequences(X_test)

# Censor the data by having a max review length (in number of words)

#use this function to load data from keras pickle instead of munging as shown above
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
#                                                      test_split=0.2)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

#example of a sentence sequence, note that lower integers are words that occur more commonly
print("x:", X_train[0]) #per observation vector of 20000 words
print("y:", y_train[0]) #positive or negative review encoding

# double check that word sequences behave/final dimensions are as expected
print("y distribution:", np.unique(y_train, return_counts=True))
print("max x word:", np.max(X_train), "; min x word", np.min(X_train))
print("y distribution test:", np.unique(y_test, return_counts=True))
print("max x word test:", np.max(X_test), "; min x word", np.min(X_test))

print("most and least popular words: ")
print(np.unique(X_train, return_counts=True))
# as expected zero is the highly used word for words not in index

#set model hyper parameters
epochs = 6
embedding_neurons = 128
lstm_neurons = 64
batch_size = 32

# Forward Pass LSTM Network

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(max_len,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size embedding
# embedding layer converts dense int input to one-hot in real time to save memory
embedded = Embedding(max_features, embedding_neurons, input_length=max_len)(sequence)
# normalize embeddings by input/word in sentence
bnorm = BatchNormalization()(embedded)

# apply forwards LSTM layer size lstm_neurons
forwards = LSTM(lstm_neurons, dropout_W=0.2, dropout_U=0.2)(bnorm)

# dropout
after_dp = Dropout(0.5)(forwards)
output = Dense(1, activation='sigmoid')(after_dp)

model_fdir_atom = Model(input=sequence, output=output)
# review model structure
print(model_fdir_atom.summary())

# Forward pass LSTM network

# try using different optimizers and different optimizer configs
model_fdir_atom.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
start_time = time.time()

history_fdir_atom = model_fdir_atom.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    validation_data=[X_test, y_test],
                    verbose=2)

end_time = time.time()
average_time_per_epoch = (end_time - start_time) / epochs
print("avg sec per epoch:", average_time_per_epoch)

# Bi-directional Atom

# based on keras tutorial: https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(max_len,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size embedding
# embedding layer converts dense int input to one-hot in real time to save memory
embedded = Embedding(max_features, embedding_neurons, input_length=max_len)(sequence)
# normalize embeddings by input/word in sentence
bnorm = BatchNormalization()(embedded)

# apply forwards LSTM layer size lstm_neurons
forwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4)(bnorm)
# apply backwards LSTM
backwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4, go_backwards=True)(bnorm)

# concatenate the outputs of the 2 LSTMs
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
after_dp = Dropout(0.5)(merged)
output = Dense(1, activation='sigmoid')(after_dp)

model_bidir_atom = Model(input=sequence, output=output)
# review model structure
print(model_bidir_atom.summary())

# Bi-directional Atom

# try using different optimizers and different optimizer configs
model_bidir_atom.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
start_time = time.time()

history_bidir_atom = model_bidir_atom.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    validation_data=[X_test, y_test],
                    verbose=2)

end_time = time.time()
average_time_per_epoch = (end_time - start_time) / epochs
print("avg sec per epoch:", average_time_per_epoch)

# run simple linear regression to compare performance

#based on grid search done by:
#https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch08/ch08.ipynb

#the tfidf vectors capture co-occurance statistics, think of each number representing how many times
#a word occured in a text and scaled by word frequency

tfidfTokenizer = Tokenizer(nb_words=max_features)
tfidfTokenizer.fit_on_sequences(X_train.tolist())
X_train_tfidf = np.asarray(tfidfTokenizer.sequences_to_matrix(X_train.tolist(), mode="tfidf"))
X_test_tfidf = np.asarray(tfidfTokenizer.sequences_to_matrix(X_test.tolist(), mode="tfidf"))

#check tfidf matrix
print(X_train_tfidf)
print(X_train_tfidf.shape, X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression

model_tfidf_reg = LogisticRegression(random_state=0, C=0.001, penalty='l2', verbose=1)
model_tfidf_reg.fit(X_train_tfidf, y_train)

from sklearn.metrics import accuracy_score
#calculate test and train accuracy
print("train acc:", accuracy_score(y_test, model_tfidf_reg.predict(X_train_tfidf)))
print("test acc:", accuracy_score(y_test, model_tfidf_reg.predict(X_test_tfidf)))

# Bi-directional rmsprop

# this example illistrate's that choice of optimizer is an important hyper-parameter for RNNs
# rmsprop gives substancially better results than atom
# in the literature these two optimizers commonly do well on RNNs

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(max_len,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size embedding
# embedding layer converts dense int input to one-hot in real time to save memory
embedded = Embedding(max_features, embedding_neurons, input_length=max_len)(sequence)
# normalize embeddings by input/word in sentence
bnorm = BatchNormalization()(embedded)

# apply forwards LSTM layer size lstm_neurons
forwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4)(bnorm)
# apply backwards LSTM
backwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4, go_backwards=True)(bnorm)

# concatenate the outputs of the 2 LSTMs
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
after_dp = Dropout(0.5)(merged)
output = Dense(1, activation='sigmoid')(after_dp)

model_bidir_rmsprop = Model(input=sequence, output=output)
# review model structure
print(model_bidir_rmsprop.summary())

# Bi-directional rmsprop

model_bidir_rmsprop.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
start_time = time.time()

history_bidir_rmsprop = model_bidir_rmsprop.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    validation_data=[X_test, y_test],
                    verbose=2)

end_time = time.time()
average_time_per_epoch = (end_time - start_time) / epochs
print("avg sec per epoch:", average_time_per_epoch)

#get weights from embedding layer and visualize

print(model_bidir_rmsprop.layers[1].get_config())
embmatrix = model_bidir_rmsprop.layers[1].get_weights()[0]
print(embmatrix.shape)

from sklearn.manifold import TSNE
topnwords = 5000
toptsne = TSNE(n_components=2, random_state=0)
tsneXY = toptsne.fit_transform(embmatrix[:topnwords, :])
tsneXY.shape

################## for graphing
# %matplotlib inline
# displaytopnwords = 250
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(tsneXY[:displaytopnwords, 0], tsneXY[:displaytopnwords, 1])
#
# for i in range(displaytopnwords):
#     ax.annotate(intToWord[i], (tsneXY[i, 0], tsneXY[i, 1]))
#
# fig.set_size_inches(25, 25)
# plt.show()
# # notice that great, most, well are clustered
# # bad don't even are clustered
# # We've learned structure in our sentiment embedding
# # neural networks give us this and other useful features for free

#guide to chart above

for i in range(displaytopnwords):
    print((tsneXY[i, 0], tsneXY[i, 1], intToWord[i]))

# Lets see what the embedding learned,
# provoking is close to great in cosine space, that's cool and definettly movie specific

from scipy.spatial.distance import euclidean

for value in np.argsort(np.apply_along_axis(lambda x: euclidean(x, embmatrix[imdbTokenizer.word_index['great'],:]),
                                            1, embmatrix))[:20]:
    print((intToWord[value], euclidean(embmatrix[value,:], embmatrix[imdbTokenizer.word_index['great'],:])))

from scipy.spatial.distance import cosine

for value in np.argsort(np.apply_along_axis(lambda x: cosine(x, embmatrix[imdbTokenizer.word_index['great'],:]),
                                            1, embmatrix))[:20]:
    print((intToWord[value], cosine(embmatrix[value,:], embmatrix[imdbTokenizer.word_index['great'],:])))

imdbTokenizer.word_index['great']

# Prep the data to do a prediction at each time step
# goal is to use this model to figure out what words cause predicted sentiment to change
# reshape to predict at every time step the review sentiment

y_train_multi = np.repeat(y_train.reshape((-1,1)), max_len, axis=1).reshape((-1,max_len,1))
y_test_multi = np.repeat(y_test.reshape((-1,1)), max_len, axis=1).reshape((-1,max_len,1))
print(y_train_multi.shape)

# Forward Pass LSTM Network multi step predict
from keras.layers.wrappers import TimeDistributed

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(max_len,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size embedding
# embedding layer converts dense int input to one-hot in real time to save memory
embedded = Embedding(max_features, embedding_neurons, input_length=max_len)(sequence)
# normalize embeddings by input/word in sentence
bnorm = BatchNormalization()(embedded)

# apply forwards LSTM layer size lstm_neurons
forwards = LSTM(lstm_neurons, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(bnorm)

# dropout
after_dp = Dropout(0.5)(forwards)
output = TimeDistributed(Dense(1, activation='sigmoid'))(after_dp)

model_fdir_multi = Model(input=sequence, output=output)
# review model structure
print(model_fdir_multi.summary())

# Forward pass LSTM network multi step predict

# try using different optimizers and different optimizer configs
model_fdir_multi.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
start_time = time.time()

history_fdir_multi = model_fdir_multi.fit(X_train, y_train_multi,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    validation_data=[X_test, y_test_multi],
                    verbose=2)

end_time = time.time()
average_time_per_epoch = (end_time - start_time) / epochs
print("avg sec per epoch:", average_time_per_epoch)

y_test_pred_mult = model_fdir_multi.predict(X_test)
y_test_pred_mult.shape

# as a sanity check look at the accuracy of the final prediction
accuracy_score(y_test_multi[:,-1,:].ravel(), y_test_pred_mult[:,-1,:].ravel() > 0.5)

print("avg starting review:", np.median(y_test_pred_mult[:,0,:]))
print("max starting review:", np.max(y_test_pred_mult[:,0,:]))
print("min starting review:", np.min(y_test_pred_mult[:,0,:]))

np.mean(y_test_pred_mult[:,-1,:].ravel())

for review in y_test_pred_mult[:,:,:][:10]:
    for word in review:
        print(word)

predDelta = y_test_pred_mult[:,1:,:] - y_test_pred_mult[:,:-1,:]
predDelta.shape

# concatenate 0.5 (neutral sentiment) to the initial dimension
predDelta = np.concatenate((np.repeat(0.5, 25000).reshape((-1,1)), predDelta.reshape((-1, max_len-1))), axis=1)
predDelta.shape

# group the predDelta value by the sequence index value in X_test
# to figure out which words cause sentiment to change the most

ascDeltaWords = [np.mean(predDelta[X_test == x]) for x in range(max_features)]

# filter out nan for words not observed in test set
ascDeltaWords = [0 if np.isnan(x) else x for x in ascDeltaWords]

print("Larget positive deltas")
count = 0
for value in np.argsort(ascDeltaWords).tolist()[::-1]:
    #filter to only look at commonly used words
    if value < 1000:
        count += 1
        print((value, intToWord[value], np.mean(predDelta[X_test == value])))
        if count > 20:
            break

print("Larget negative deltas")
count = 0
for value in np.argsort(ascDeltaWords).tolist():
    #filter to only look at commonly used words
    if value < 1000:
        count += 1
        print((value, intToWord[value], np.mean(predDelta[X_test == value])))
        if count > 20:
            break

################## for graphing

# %matplotlib inline
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(tsneXY[posToNegWordList, 0], tsneXY[posToNegWordList, 1])
#
# for i in posToNegWordList:
#     ax.annotate(intToWord[i], (tsneXY[i, 0], tsneXY[i, 1]))
#
# fig.set_size_inches(25, 25)
# plt.show()
# # notice that we have two very distinct clusters in the embedding as one would expect

from gensim.models import Word2Vec

#get pre trained word2vec from google:
#https://doc-0k-4g-docs.googleusercontent.com/docs/securesc/gnqvgap6hjncpd3b10i2tv865io48jas/hmjtdgee48c14e1parufukrpkb8urra5/1463018400000/06848720943842814915/09676831593570546402/0B7XkCwpI5KDYNlNUTTlSS21pQmM?e=download&nonce=4l49745nmtine&user=09676831593570546402&hash=i2qe9mshan4mesl112ct9bu1tj9kr1hq

googlew2v = Word2Vec.load_word2vec_format('./googleword2vec/GoogleNews-vectors-negative300.bin', binary=True)

# get word vectors for words in my index
googleVecs = []
for value in range(max_features):
    try:
        googleVecs.append(googlew2v[intToWord[value]])
    except:
        googleVecs.append(np.random.uniform(size=300))

googleVecs = np.array(googleVecs)

print(googleVecs)
print(googleVecs.shape)

# Bi-directional google

# this example tests if using pretrained embeddings will improve performance
# relative to starting with random embeddings

# this is the placeholder tensor for the input sequences
sequence = Input(shape=(max_len,), dtype='int32')
# this embedding layer will transform the sequences of integers
# into vectors of size embedding
# embedding layer converts dense int input to one-hot in real time to save memory
embedded = Embedding(max_features, 300, input_length=max_len, weights=[googleVecs])(sequence)
# normalize embeddings by input/word in sentence
bnorm = BatchNormalization()(embedded)

# apply forwards LSTM layer size lstm_neurons
forwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4)(bnorm)
# apply backwards LSTM
backwards = LSTM(lstm_neurons, dropout_W=0.4, dropout_U=0.4, go_backwards=True)(bnorm)

# concatenate the outputs of the 2 LSTMs
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
after_dp = Dropout(0.5)(merged)
output = Dense(1, activation='sigmoid')(after_dp)

model_bidir_google = Model(input=sequence, output=output)
# review model structure
print(model_bidir_google.summary())

# Bi-directional google

model_bidir_google.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
start_time = time.time()

history_bidir_google = model_bidir_google.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    validation_data=[X_test, y_test],
                    verbose=2)

end_time = time.time()
average_time_per_epoch = (end_time - start_time) / epochs
print("avg sec per epoch:", average_time_per_epoch)

test = "I would want to see this movie."

test = imdbTokenizer.texts_to_sequences([test])

from keras.preprocessing import sequence
test = sequence.pad_sequences(test, maxlen=max_len)

test

model_bidir_rmsprop.predict(test)
