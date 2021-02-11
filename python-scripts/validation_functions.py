
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from pandas import read_csv
from matplotlib import pyplot

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Perceptron
from sklearn import datasets, svm, metrics
from sklearn import preprocessing
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import collections
import pickle
import math
import os

def cross_validation_functions(model, input, output):
   kfold = StratifiedKFold(n_splits=10, random_state=1)
   cv_results = cross_val_score(model, input, output, cv=kfold, scoring='accuracy')
   y_pred = cross_val_predict(model, input, output, cv=10)
   conf_mat = confusion_matrix(output, y_pred)
   mean = cv_results.mean()
   std = cv_results.std()
   return ({
      'cv_results': cv_results,
      'conf_mat': conf_mat,
      'mean': mean,
      'std': std
   })

def roc_validation_function(model,y_true,y_pred):
   roc_auc = roc_auc_score(y_true, y_pred)
   ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
   pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline (random guess)')
   pyplot.xlabel('False Positive Rate')
   pyplot.ylabel('True Positive Rate')
   return roc_auc

def confusion_matrix_function(y_true, y_pred, labels):
   confusion_matrix(y_true, y_pred, labels)
   print(confusion_matrix)
   return confusion_matrix

def regression(y_test, pred_results, model):
   # st.write('R2 ', r2_score(y_test, pred_results))
   # st.write('MSE ', mean_squared_error(y_test, pred_results))
   # st.write('MAE ', mean_absolute_error(y_test, pred_results))
   result = {
      'R2': r2_score(y_test, pred_results),
      'MSE': mean_squared_error(y_test, pred_results),
      'MAE': mean_absolute_error(y_test, pred_results)
   }
   # plot_tree(model, filled=True)
   # plt.show()
   # st.show()
   return(result)