from airbnbRegression import airbnb_regression
from airbnbClassifier import airbnb_classifier
import streamlit as st

def airbnbdataset():
    ml_type = ['------', 'Regression', 'Classification']
    option = st.sidebar.selectbox('ML Type', ml_type)
    if option == 'Regression':
        airbnb_regression()
    elif option == 'Classification':
        airbnb_classifier()
    else:
    	pass
    return

