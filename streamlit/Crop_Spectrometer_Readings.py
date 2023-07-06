import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SpeCROPmeter Readings", page_icon="🌱")

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.feature_selection import mutual_info_regression
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, roc_auc_score


st.markdown("""
# Spectrometer Readings 🌾🌻🎃🍆
"""
)

#Load the models
#with open("./models/logreg_model.pkl", "rb") as file:
#    logreg_model = pickle.load(file)
with open("./pickle_objects/initdf.pkl", "rb") as file:
    df = pickle.load(file)
    
st.markdown("### The Data:")
st.markdown('''
A total of 2650 grains of barley, chickpea and sorghum cultivars were scanned using the SCIO, a recently released miniaturized NIR spectrometer. For each cultivar, 50 grains were randomly selected for scanning.

Column 740 to 1070 corresponds to the SCIO wavelengths, measured in nanometers (331 variables).
Here is a sample of the initial un-preprocessed dataframe:
''')
st.dataframe(df.iloc[[1,50,100,150,200,250]],hide_index=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown('''
In the dataframe above, we see crops named in shorthand. For example, for predictors that start with "HB" are cultivars of Barley.
''')
with col2:
    st.image("https://cdn-prod.medicalnewstoday.com/content/images/articles/295/295268/barley-grains-in-a-wooden-bowl.jpg")


       


