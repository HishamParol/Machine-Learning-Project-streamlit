from sklearn import preprocessing, datasets, metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import np_utils

import models
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import collections
import pickle
import time
import math
import os

dataset_name = "Robot Dataset"
labels = ['S1', 'S2', 'S3', 'S4']

@st.cache
def load_data():
	data = pd.read_csv('../xlsx-csv/RobotDataset/sensor_readings_24.data')
	# Code for preprocessing
	# data = 24 features dataset
	processed_data = pd.DataFrame(data=None, columns=['S1', 'S2', 'S3', 'S4', 'OUTPUT'])
	processed_data['S1'] = data.loc[:, ['S11', 'S12', 'S13', 'S14', 'S15']].min(axis=1)
	processed_data['S2'] = data.loc[:, ['S18', 'S19', 'S20']].min(axis=1)
	processed_data['S3'] = data.loc[:, ['S5', 'S6', 'S7', 'S8', 'S9']].min(axis=1)
	processed_data['S4'] = data.loc[:, ['S23', 'S24']].min(axis=1)
	processed_data['OUTPUT'] = data['OUTPUT']
	X = processed_data[labels]
	Y = processed_data['OUTPUT']
	return data, X, Y


def preprocess(Y):
	le = preprocessing.LabelEncoder()
	le.fit(Y)
	list(le.classes_)
	Yt = le.transform(Y)
	return Yt

def robotdataset():
	data, X, Y = load_data()
	Y = preprocess(Y)
	processed_Y = pd.DataFrame(data=Y, columns=["Output"])
	processed_data = pd.concat([X, processed_Y], axis=1)
	ds_name = "Tf_keras"
	disp_data = st.sidebar.radio("Data", ("Raw Data", "Processed Data"))
	if disp_data == "Raw Data":
		st.subheader("Raw Data")
		st.write(data)
	else:
		st.subheader("Processed Data")
		st.write(processed_data)
	model = ["------", "Naive Bayes", "Decision Tree", "Perceptron", "MLP", "SVM", "XGBoost", "Keras NN"]
	option = st.sidebar.selectbox('Machine Learning Model', model)
	with st.spinner('Training the model'):
		if option == "Naive Bayes":
			model = models.transcendence_naive(X, Y, labels, dataset_name)
		if option == "Decision Tree":
			model = models.transcendence_decision_tree(X, Y, labels, dataset_name, random_state=5, criterion='entropy', ccp_alpha=0.005, min_samples_split=2)
		if option == "Perceptron":
			model = models.transcendence_perceptron(X, Y, labels, dataset_name, eta0=0.1, random_state=0, max_iter=100)
		if option == "MLP":
			model = models.transcendence_mlp(X, Y, labels, dataset_name, random_state=0, learning_rate=0.05, activation='logistic', hidden_layer_sizes=(6,), max_iter=500)
		if option == "SVM":
			model = models.transcendence_svm(X, Y, labels, dataset_name, gamma=0.001)
		if option == "XGBoost":
			model = models.transcendence_xgboost(X, Y, labels, dataset_name)
		if option == "Keras NN":
			ip_dim=4
			op_dim=4
			hlayers=35
			batch_size=32
			epochs=200
			val_split=0.30
			model = models.tfneuralnetwork(X, Y, labels, ip_dim, op_dim, dataset_name, hlayers, batch_size, epochs, val_split)
		else:
			pass
	return
