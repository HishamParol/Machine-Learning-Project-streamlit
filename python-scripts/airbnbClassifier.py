from sklearn import preprocessing, datasets, metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import np_utils
from validation_functions import cross_validation_functions
from tensorflow_nn import tfneuralnetwork
from roc_curve import roc_curve_plot
from confusion_matrix_plot import confusion_matrix_plot
import seaborn as sns

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

dataset_name = "Airbnb Dataset"
labels = [0, 1, 2]
labels1 = [0, 1, 2, 3, 4, 5]

# Function to pre-process data for predicting the best hotels based on the review_count
def airbnb_load_preprocess():
	airbnb_data = pd.read_csv('../xlsx-csv/NYAirbnb/AB_NYC_2019.csv')
	raw_data = airbnb_data
	airbnb_data.dropna(subset=["last_review"], inplace=True)
	years = pd.DatetimeIndex(airbnb_data['last_review']).year
	sns.catplot(x="room_type", y="price", data=airbnb_data)
	st.pyplot()
	sns.catplot(x="neighbourhood", y="price", data=airbnb_data)
	st.pyplot()
	sns.catplot(x="room_type", y="number_of_reviews", data=airbnb_data)
	st.pyplot()
	for i in range(len(airbnb_data)):
		if years[i] < 2015:
			airbnb_data.drop(airbnb_data.index[i], inplace=True)
	airbnb_data = airbnb_data.reset_index(drop=True)
	X = airbnb_data[['id', 'host_id', 'neighbourhood_group', 'neighbourhood', 'room_type', 'price']]
	# convert categorigcal variables to numeric via one-hot encoding
	X = pd.get_dummies(X)
	# divide number_of_reviews into bins to get different classifications
	airbnb_data['reviews_bins'] = pd.cut(x=airbnb_data['number_of_reviews'], bins=[0, 30, 50, 650], labels=[0, 1, 2])
	Y = airbnb_data.reviews_bins
	return X, Y

# The airbnb classifier function
def airbnb_classifier():
	X, Y = airbnb_load_preprocess()
	model = ["------", "Naive Bayes", "Decision Tree", "Perceptron", "MLP", "XGBoost"]
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
		else:
			pass
	return

# # Function to pre-process data to predict price. Not used as it was not having any better result in classification
# def airbnb_price_preprocess():
# 	airbnb_data = pd.read_csv('../xlsx-csv/NYAirbnb/AB_NYC_2019.csv')
# 	airbnb_data.dropna(subset=["last_review"], inplace=True)
# 	raw_data = airbnb_data
# 	airbnb = airbnb_data[['id', 'host_id', 'neighbourhood_group','neighbourhood', 'room_type', 'price', 'number_of_reviews','last_review']]
# 	years = pd.DatetimeIndex(airbnb_data['last_review']).year
# 	sns.catplot(x="room_type", y="price", data=airbnb)
# 	st.pyplot()
# 	sns.catplot(x="neighbourhood", y="price", data=airbnb)
# 	st.pyplot()
# 	sns.catplot(x="neighbourhood", y="number_of_reviews", data=airbnb)
# 	st.pyplot()
# 	for i in range(len(airbnb_data)):
# 		if years[i] < 2015:
# 			airbnb_data.drop(airbnb_data.index[i], inplace=True)
# 	airbnb_data = airbnb_data.reset_index(drop=True)
# 	X = airbnb_data[['id', 'host_id', 'neighbourhood_group', 'neighbourhood', 'room_type']]
# 	# convert categorigcal variables to numeric via one-hot encoding
# 	X = pd.get_dummies(X)
# 	# divide number_of_reviews into bins to get different classifications
# 	airbnb_data['reviews_bins'] = pd.cut(airbnb_data['price'], bins=[-1, 110, 220, 350, 700, 2000, 10000],labels=[0,1,2,3,4,5])
# 	Y = airbnb_data.reviews_bins
# 	return(X, Y)

# # SMOTE function used to balance unbalanced dataset - however this resulted is significantly lower accuracy, so not used
# def airbnb_smote():
#     dataset_name = "Airbnb Dataset"
#     airbnb_data = pd.read_csv('../xlsx-csv/NYAirbnb/AB_NYC_2019.csv')
#     airbnb_data.dropna(subset=["last_review"], inplace=True)
#     years = pd.DatetimeIndex(airbnb_data['last_review']).year
#     # print(len(airbnb_data))
#     for i in range(len(airbnb_data)):
#         if years[i] < 2016:
#             airbnb_data.drop(airbnb_data.index[i], inplace=True)
#     airbnb_data = airbnb_data.reset_index(drop=True)

#     X = airbnb_data[['id', 'host_id', 'neighbourhood_group', 'neighbourhood', 'room_type', 'price']]
#     # convert categorigcal variables to numeric via one-hot encoding
#     X = pd.get_dummies(X)
#     # divide number_of_reviews into bins to get different classifications
#     airbnb_data['reviews_bins'] = pd.cut(x=airbnb_data['number_of_reviews'], bins=[0, 30, 50, 650], labels=[0, 1, 2])
#     Y = airbnb_data.reviews_bins
#     airbnb_data=airbnb_data.dropna()
#     airbnb_data.reset_index(drop=True,inplace=True)
#     # SMOTE number of neighbours
#     k = 5
#     seed = 47
#     sm_airbnb = SMOTE(sampling_strategy='auto',k_neighbors=k, random_state=seed)
#     X_res, Y_res = sm_airbnb.fit_resample(X, Y)
#     plt.title('Dataset balanced with synthetic or SMOTEd data', k 'neighbours.')
#     plt.xlabel('X')
#     plt.ylabel('Y')

#     plt.scatter(X_res.iloc[:, 0],X_res.iloc[:, 1], marker='o', c=Y_res,
#                s=25, edgecolor='k', cmap=plt.cm.coolwarm)
#     plt.show()
#     return (X_res,Y_res)
