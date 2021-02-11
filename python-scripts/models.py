from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from precision_recall import precision_recall

from sklearn import svm

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from keras.utils import np_utils
from xgboost import XGBClassifier, plot_tree

from validation_functions import cross_validation_functions, regression
from roc_curve import roc_curve_plot, roc_plot_yellowbrick, keras_roc_auc_plot
from confusion_matrix_plot import confusion_matrix_plot, keras_confusion_matrix_plot
from kfold_keras import kfold_keras

import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

##########################################################################
# Defining the models
def transcendence_naive(X, Y, labels, dataset_name):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = GaussianNB()
	model_name = "Naive Bayes"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def transcendence_decision_tree(X, Y, labels, dataset_name, random_state, criterion, ccp_alpha, min_samples_split):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = DecisionTreeClassifier(random_state=random_state, criterion=criterion, ccp_alpha=ccp_alpha, min_samples_split=min_samples_split)
	model_name = "Decision Tree"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels) 
	st.subheader(model_name)
	dot_data  = export_graphviz(model, filled=True, rounded=True, out_file=None)
	st.graphviz_chart(dot_data)
	return model

def transcendence_perceptron(X, Y, labels, dataset_name, eta0, random_state, max_iter):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = Perceptron(eta0=eta0, random_state=random_state, max_iter=max_iter)
	model_name = "Perceptron"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def transcendence_mlp(X, Y, labels, dataset_name, random_state, activation, hidden_layer_sizes, max_iter, learning_rate):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = MLPClassifier(random_state=random_state, activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate)
	model_name = "MLP"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def transcendence_svm(X, Y, labels, dataset_name, gamma):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = svm.SVC(gamma=gamma)
	model_name = "SVM"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def transcendence_xgboost(X, Y, labels, dataset_name):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = XGBClassifier(objective="binary:logistic", random_state=42)
	model_name = "XGBoost"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def tfneuralnetwork(X, Y, labels, ip_dim, op_dim, dataset_name, hlayers, batch_size, epochs, val_split):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	ip_data=X_train
	op_data=np_utils.to_categorical(y_train)
	op_dim=4
	# op_data=y_train
	model_name = "Keras NN"
	model = Sequential()
	model.add(Dense(hlayers, input_dim=ip_dim, activation='relu'))
	model.add(Dense(op_dim, activation='relu'))
	tensorboard = TensorBoard(log_dir="logs\{}".format(dataset_name))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(ip_data, op_data, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks=[tensorboard])
	kfold_cross_validation_keras(X_train, y_train, model, labels)
	y_test_pred=model.predict(X_test)
	y_test_pred_decoded = y_test_pred.argmax(1)
	test_keras_nn(X_test, y_test, y_test_pred_decoded, model, labels)
	# keras_roc_auc_plot(model, labels, X_train, y_train, X_test, y_test)
	# keras_confusion_matrix_plot(model, y_test, y_test_pred, dataset_name, model_name, labels)
	# keras_confusion_matrix_plot(model, y_test, y_test_pred_decoded, dataset_name, model_name, labels)
	return model

def transcendence_ols(X, y):
	model = sm.OLS(y, X).fit()
	predictions = model.predict(X)
	st.write("OLS")
	summary = model.summary()
	print(summary)
	# st.write(summary)
	return model

def transcendence_dt_regression(X, y, max_depth):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	model = DecisionTreeRegressor(max_depth=max_depth)
	model.fit(X_train, y_train)
	pred_results = model.predict(X_test)
	result = regression(y_test, pred_results, model)
	st.write("Evaluation Metrics ", result)
	dot_data = export_graphviz(model, filled=True, rounded=True, out_file=None)
	st.graphviz_chart(dot_data)
	return

# Scikit validation and testing
def scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels):
	model.fit(X_train, y_train)
	kfold_cross_validation_scikit(X_train, y_train, model)
	y_test_pred=model.predict(X_test)
	test_scikit_models(y_test, y_test_pred)
	if dataset_name == "Robot Dataset":
		plotting_ROC_curve_yellowbrick(model, labels, X_train, y_train, X_test, y_test)
	else:
		precision_recall_curve(X, Y, model, model_name)
	confusion_matrix_plot(model, X_test, y_test, dataset_name, model_name, labels)
	return

##########################################################################
# Validation of models
def kfold_cross_validation_scikit(X_train, y_train, model):
	result = cross_validation_functions(model, X_train, y_train)
	st.subheader("Validation Result - KFold")
	st.write("Accuracy: %.2f" % (result['mean']*100), "%")
	st.write("Standard Deviation: %.2f" % (result['std']*100))
	st.write("Confusion Matrix:\n", result['conf_mat'])
	return

def kfold_cross_validation_keras(X_train, y_train, model, labels):
	keras_cv = kfold_keras(X_train, y_train, labels)
	st.subheader("Validation Result - KFold")
	st.write("Accuracy: %.2f" % (keras_cv['mean']), "%")
	st.write("Standard Deviation: %.2f" % (keras_cv['std']))
	return

##########################################################################
# Testing the models
def test_scikit_models(y_test, y_test_pred):
	st.subheader("Test Result")
	accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)
	st.write("Accuracy = %.2f" % (accuracy*100), "%")      
	st.write("Confusion Matrix :\n", confusion_matrix(y_true=y_test, y_pred=y_test_pred))
	return

def test_keras_nn(X_test, y_test, y_test_pred, model, labels):
	st.subheader("Test Result")
	# y_test_encoded = np_utils.to_categorical(y_test)
	# keras_cv = kfold_keras(X_test, y_test, labels)
	loss, accuracy = model.evaluate(X_test, np_utils.to_categorical(y_test))
	st.write("Accuracy = %.2f" % (accuracy*100), "%")
	st.write("Confusion Matrix :\n", confusion_matrix(y_true=y_test, y_pred=y_test_pred, labels=labels))
	return

def classification_report(y_test, y_class_report):
	# y_class_report = y_test_pred_decoded if option == "Keras NN" else y_test_pred
	cs_report = classification_report(y_true=y_test, y_pred=y_class_report, output_dict=True)
	cs_report_df = pd.DataFrame(cs_report).transpose()
	st.subheader("Classification Report :\n")
	st.dataframe(cs_report_df)

##########################################################################
# calling the ROC plot
def plotting_ROC_curve_yellowbrick(model, labels, X_train, y_train, X_test, y_test):
	st.subheader('ROC Curve')
	roc_plot_yellowbrick(model, labels, X_train, y_train, X_test, y_test)
	return

def plotting_ROC_curve_keras():
	
	return

def plotting_split_ROC(option, X, Y, model, label_classes):
	if option == "SVM" or option == "Perceptron":
		roc_curve_plot(X, Y, model, label_classes, parameter=1)
	else:
		roc_curve_plot(X, Y, model, label_classes, parameter=0)
	return

def precision_recall_curve(X, Y, model, model_name):
	st.subheader("Precision-Recall")
	if model_name == "SVM" or model_name == "Perceptron":
		precision_recall(X, Y, model, parameter=1)
	else:
		precision_recall(X, Y, model, parameter=0)
	return
