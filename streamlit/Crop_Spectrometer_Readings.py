import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="SpeCROPmeter Readings", page_icon="🌱")

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.feature_selection import mutual_info_regression
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, roc_auc_score


st.markdown("""
# Spectrometer Readings 🌾🎃🍆
"""
)

#Load the models
#with open("./models/logreg_model.pkl", "rb") as file:
#    logreg_model = pickle.load(file)
with open("./pickle_objects/initdf.pkl", "rb") as file:
    df = pickle.load(file)
    
st.markdown('''
A total of 2650 grains of barley, chickpea and sorghum cultivars were scanned using the SCIO, a recently released miniaturized NIR spectrometer. For each cultivar, 50 grains were randomly selected for scanning.

Column 740 to 1070 corresponds to the SCIO wavelengths, measured in nanometers (331 variables).

This dataset was retrieved from kaggle and can be found [here](https://www.kaggle.com/datasets/fkosmowski/crop-varietal-identification-with-scio)''')    
    
st.markdown("## The Data: 🔍📊")
st.markdown('''Here is a sample of the initial un-preprocessed dataframe:
''')
st.dataframe(df.iloc[[1,50,100,150,200,250]],hide_index=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown('''
In the dataframe above, we see crops named in shorthand. For example, predictors that start with "HB" are cultivars of Barley. 
''')
with col2:
    st.image("https://cdn-prod.medicalnewstoday.com/content/images/articles/295/295268/barley-grains-in-a-wooden-bowl.jpg")
    
st.markdown(''' 
Let's have an idea of the shape of the readings. We plot the mean of the readings for each cultivar type with the code below:''')
st.code('''
plt.figure(figsize=(10, 8))
for i in range(0,mean_df.shape[0]):
    Y = mean_df.iloc[i]
    X = mean_df.columns.astype(int)
    plt.plot(X, Y, label=Y.name)

plt.xlabel('Wavelength')
plt.ylabel('Mean reading')
plt.title('Mean reading for each crop type')
plt.savefig('./plots/eda_mean_read.png')

plt.legend(loc=(1, 0.25))
plt.show()
''')
st.image("./plots/eda_mean_read.png")

st.markdown('''## t-SNE: 🌌🗺️
#### Visualising the data in a lower dimensional space
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique used for visualizing high-dimensional data in a lower-dimensional space. It creates a map where similar data points in the original space are closer together. By optimizing the positions of the points, t-SNE reveals underlying patterns and structures in complex datasets.''')

with open("./pickle_objects/tsne2d.pkl", "rb") as file:
    tsne2df = pickle.load(file)
with open("./pickle_objects/tsne3d.pkl", "rb") as file:
    tsne3df = pickle.load(file)    

col3, col4 = st.columns([1,5])
with col3:
    
    genre = st.radio(
    "**Dimensions**",
    (2,3),
    index=1)
with col4:
    if genre:
        if genre == 2:
            fig, ax = plt.subplots()
            plt.scatter(tsne2df['Dimension 1'], tsne2df['Dimension 2'], c=tsne2df['ePred'])
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('t-SNE Visualization')
            st.pyplot(fig)
        else:
            fig = go.Figure(data=go.Scatter3d(
                x=tsne3df['Dimension 1'],
                y=tsne3df['Dimension 2'],
                z=tsne3df['Dimension 3'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=tsne3df['ePred'],
                    colorscale='Viridis',
                    opacity=0.8
                )
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'
                ),
                title='t-SNE Visualization (3D), try moving the plot with your mouse!'
            )
            st.plotly_chart(fig)
            
st.markdown('''
While the plots produced certainly looks very intriguing and cool, separation between the various classes are poor. Perhaps the plot is cluttered with too many scatter points and classes. This is what Chat-GPT had to say:
''')
st.code('''
When t-SNE dimension-reduced plots do not exhibit good separation between classes, it indicates that the 
data points from different classes are not clearly distinguishable in the lower-dimensional space. This
lack of separation can occur due to various reasons. One possibility is that the data distributions of 
different classes overlap, making it challenging for t-SNE to effectively separate them. Another reason 
could be the presence of high intra-class variability, where data points within the same class are 
scattered and mixed together. Additionally, it's possible that the information contained in the original 
high-dimensional data is not fully captured by the t-SNE transformation, resulting in reduced 
discriminative power.''', language=None)

st.markdown('''Before we explore alternatives in dimension reduction techniques, let us first train a model using all of the original variables to establish a baseline''')
st.markdown("## Baseline Results")
st.markdown("Let's first standardize the relevant numerical values")
st.code('''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_values = scaler.fit_transform(base_df.drop('ePred',axis=1))
base_df_standardized = pd.DataFrame(standardized_values, columns=base_df.drop('ePred',axis=1).columns)
base_df_standardized['ePred'] = base_df.ePred
base_df_standardized
''')
st.markdown("Let's use XGBoost as a baseline model with a simple training procedure:")
st.code('''
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

feature_cols = base_df_standardized.drop('ePred',axis=1).columns
target = base_df['ePred']

X_train, X_test, y_train, y_test = train_test_split(base_df_standardized[feature_cols], target, test_size=0.3, random_state=42)

print("Fitting model...")
base_model = xgb.XGBClassifier(n_estimators = 50, n_jobs = -1)  
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)
print("Done!")
''')
st.markdown("Obtaining these results:")
col5, col6 = st.columns(2)
with col5:
    st.image("./plots/xgb_init_featimp.png")
with col6:    
    st.image("./plots/xgb_init_cm.png")
st.markdown("We achieved a commendable score of 28.6%, and XGBoost identified the reading at 770nm as particularly significant. It's worth noting that we are dealing with 23 distinct classes, so these results are quite promising compared to a completely random guess, which would only yield approximately 4% accuracy.")    
            
            


       


