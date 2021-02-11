# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

import streamlit as st
import numpy
	
def kfold_keras(X, Y, labels):
	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)
	X = X.to_numpy()
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	cvscores = []
	with st.spinner('Validating the model using KFold cross validation'):
		bar = st.progress(0)
		i = 1
		for train, test in kfold.split(X, Y):
			# create model
			Y = label_binarize(Y, classes=labels)
			model = Sequential()
			model.add(Dense(35, input_dim=4, activation='relu'))
			# model.add(Dense(4, activation='relu'))
			model.add(Dense(4, activation='sigmoid'))
			# Compile model
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			# Fit the model
			model.fit(X[train], Y[train], epochs=100, batch_size=32, verbose=0)
			# evaluate the model
			scores = model.evaluate(X[test], Y[test], verbose=0)
			cvscores.append(scores[1] * 100)
			bar.progress(i*10)
			i += 1
		bar.empty()
	return ({'cvscores': cvscores,
			'mean': numpy.mean(cvscores),
			'std': numpy.std(cvscores)})