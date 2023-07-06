import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

'''
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
'''

st.set_page_config(page_title="Tumor Classification")

st.markdown("""
# Analysing Crop Spectrometer Readings
"""
)

#Load the models
#with open("./models/logreg_model.pkl", "rb") as file:
#    logreg_model = pickle.load(file)

       
    
col1, col2 = st.columns(2)
with col1:
    st.write("Hi!")
with col2:
    st.write("Crop")

