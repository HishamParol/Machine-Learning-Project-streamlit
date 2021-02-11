# The home page or the first page
import streamlit as st
import pandas as pd
import numpy as np
import statistics
import time

# importing plotly
import plotly.figure_factory as ff
import plotly.express as px

# import Robot functions
from robotdataset import robotdataset
# import Airbnb functions
from airbnb import airbnbdataset

pd.options.plotting.backend = "plotly"

dataset = ['------', 'Robot', 'Airbnb']

def side_bar():
    st.sidebar.title('Transcendence')
    option = st.sidebar.selectbox('Dataset', dataset)
    if option == 'Robot':
        robotdataset()
    if option == 'Airbnb':
        airbnbdataset()
    else:
        pass
    return

def main(): 
    side_bar()
    
if __name__ == '__main__':
    main()