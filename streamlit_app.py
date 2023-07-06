import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="SpeCROPmeter Readings", page_icon="🌱")

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
st.image("./plots/eda_mean_read_2.png")

st.markdown('''## t-SNE: 🌌🗺️
#### Visualising the data in a lower dimensional space
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique used for visualizing high-dimensional data in a lower-dimensional space. It creates a map where similar data points in the original space are closer together. By optimizing the positions of the points, t-SNE reveals underlying patterns and structures in complex datasets.''')

with open("./pickle_objects/tsne2d.pkl", "rb") as file:
    tsne2df = pickle.load(file)
with open("./pickle_objects/tsne3d.pkl", "rb") as file:
    tsne3df = pickle.load(file)    

col3, col4 = st.columns([1,5])
with col3:
    st.markdown("**Select Dimension:**")
    d1 = st.radio(
    "**Dimensions**",
    (2,3),
    index=1,
    label_visibility="collapsed")
with col4:
    if d1:
        if d1 == 2:
            fig, ax = plt.subplots()
            plt.scatter(tsne2df['Dimension 1'], tsne2df['Dimension 2'],
                        c=tsne2df['ePred'])
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
When t-SNE dimension-reduced plots do not exhibit good separation between 
classes, it indicates that the data points from different classes are not 
clearly distinguishable in the lower-dimensional space. This lack of 
separation can occur due to various reasons. One possibility is that the 
data distributions of different classes overlap, making it challenging for 
t-SNE to effectively separate them. Another reason could be the presence 
of high intra-class variability, where data points within the same class 
are scattered and mixed together.''', language=None)

st.markdown('''Before we explore alternatives in dimension reduction techniques, let us first train a model using all of the original variables to establish a baseline''')
st.markdown("## Baseline Results 🧱📊")
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
col5, col6 = st.columns([6,5])
with col5:
    st.image("./plots/xgb_init_featimp.png")
with col6:    
    st.image("./plots/xgb_init_cm.png")
st.markdown("We achieved a commendable score of 28.6%, and XGBoost identified the reading at 770nm as particularly significant. It's worth noting that we are dealing with 23 distinct classes, so these results are quite promising compared to a completely random guess, which would only yield approximately 4% accuracy.") 

st.markdown("## PCA 🧩📊")
st.markdown("Principal Component Analysis (PCA) is a popular dimensionality reduction technique used to simplify complex datasets while retaining their essential structure. The main objective of PCA is to transform a high-dimensional dataset into a lower-dimensional space by identifying the principal components that capture the most significant variations in the data.")
st.markdown("### Differentiating the data")
st.code('''
from scipy.signal import savgol_filter

X_dx = savgol_filter(base_df_standardized[feature_cols], 
                     window_length=25, 
                     polyorder = 5, 
                     deriv=1)
''')
st.markdown('''
Taking the first derivative of the data enables us to correct for baseline differences in the scans, and highlight the major sources of variation between the different scans. Numerical derivatives are generally unstable, so we use the smoothing filter implemented in scipy.signal, [savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) to smooth the derivative data out.

Idea acknowledgement: https://nirpyresearch.com/classification-nir-spectra-principal-component-analysis-python/

Let's plot the differentiated data for a sanity check:
''')
st.image("./plots/differentiate.png")

st.markdown('''
From the article referenced [here](https://nirpyresearch.com/classification-nir-spectra-principal-component-analysis-python/), the author suggests that "a good rule of thumb is to choose the number of principal components by looking at the cumulative variance of the decomposed (original) spectral data." So let's take a look at plots of explained and cumulative variance.
''')
st.image("./plots/cumulvarpercent.png")
st.markdown("If we follow that rule, we should only use the first component as it explains almost 100% of the original data. I decide to use the first 6 principal components instead, which explains around 90% of the first derivative data. In a 23-parameter dataset, it is unlikely that only the first principal component of PCA will capture all the essential information. While the first PC accounts for the largest variance in the data, it may not necessarily encompass all the meaningful patterns and relationships present in the dataset. Subsequent PCs capture additional variations orthogonal to the previous ones, providing a more comprehensive representation of the data.")

with open("./pickle_objects/pcscores.pkl", "rb") as file:
    Xt2 = pickle.load(file)
with open("./pickle_objects/ytrain.pkl", "rb") as file:
    y_train = pickle.load(file)    
with open("./pickle_objects/label_mapping.pkl", "rb") as file:
    label_mapping = pickle.load(file)   

col7, col8 = st.columns([1,5])
        
with col7:
    st.markdown("**Select Dimension:**")
    d2 = st.radio(
    "**Dimensions**",
    (2,3),
    index=1,
    label_visibility="collapsed",
    key=2)
with col8:
    if d2:
        if d2 == 2:
            st.image('./plots/2dpca.png')
        else:
            pc1_scores = Xt2[:, 0]
            pc2_scores = Xt2[:, 1]
            pc3_scores = Xt2[:, 2]

            fig = go.Figure(data=go.Scatter3d(
                x=pc1_scores,
                y=pc2_scores,
                z=pc3_scores,
                mode='markers',
                marker=dict(
                    size=5,
                    color=y_train,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(
                        title='Class Labels',
                        titleside='right',
                        tickvals=list(label_mapping.keys()),
                        ticktext=list(label_mapping.values())
                    )
                )
            ))

            # Set axis labels and title
            fig.update_layout(scene=dict(
                xaxis_title='PC1 Scores',
                yaxis_title='PC2 Scores',
                zaxis_title='PC3 Scores',
                ),
                title='Score Plot of PC1 vs PC2 vs PC3'
            )
            st.plotly_chart(fig)
            
            
st.markdown("Regrettably, upon examining the first three components, we observe limited distinction among them. Nevertheless, in the 3D visualization, we do identify four objects resembling cylinders, while one specific column appears to predominantly contain yellow scatter points.")

st.markdown("## Classification using PCs as input 🔢🎯")
st.markdown("Let's experiment with a range of models and see how they fare with the new principal components we generated.")
st.code('''
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score

results = []
confusion_matrices = []
roc_auc = []

print("Fitting XGBoost...")
xgb_model = xgb.XGBClassifier()
xgb_model.fit(pca_Xtrain, pca_ytrain)
xgb_accuracy = xgb_model.score(pca_Xtest, pca_ytest)
xgb_y_pred = xgb_model.predict(pca_Xtest)
xgb_conf_matrix = confusion_matrix(pca_ytest, xgb_y_pred)
xgb_roc_auc = roc_auc_score(pca_ytest, xgb_model.predict_proba(pca_Xtest), multi_class='ovr')

roc_auc.append(xgb_roc_auc)
results.append(xgb_accuracy)
confusion_matrices.append(xgb_conf_matrix)


print("Fitting RandomForest...")
rf_model = RandomForestClassifier()
rf_model.fit(pca_Xtrain, pca_ytrain)
rf_accuracy = rf_model.score(pca_Xtest, pca_ytest)
rf_y_pred = rf_model.predict(pca_Xtest)
rf_conf_matrix = confusion_matrix(pca_ytest, rf_y_pred)
rf_roc_auc = roc_auc_score(pca_ytest, rf_model.predict_proba(pca_Xtest), multi_class='ovr')

roc_auc.append(rf_roc_auc)
results.append(rf_accuracy)
confusion_matrices.append(rf_conf_matrix)


print("Fitting Logistic Regression...")
lr_model = LogisticRegression()
lr_model.fit(pca_Xtrain, pca_ytrain)
lr_accuracy = lr_model.score(pca_Xtest, pca_ytest)
lr_y_pred = lr_model.predict(pca_Xtest)
lr_conf_matrix = confusion_matrix(pca_ytest, lr_y_pred)
lr_roc_auc = roc_auc_score(pca_ytest, lr_model.predict_proba(pca_Xtest), multi_class='ovr')

roc_auc.append(lr_roc_auc)
results.append(lr_accuracy)
confusion_matrices.append(lr_conf_matrix)


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    print(f"Fitting SVM ({kernel.capitalize()})...")
    svm_model = SVC(kernel=kernel, probability=True)
    svm_model.fit(pca_Xtrain, pca_ytrain)
    svm_accuracy = svm_model.score(pca_Xtest, pca_ytest)
    svm_y_pred = svm_model.predict(pca_Xtest)
    svm_conf_matrix = confusion_matrix(pca_ytest, svm_y_pred)
    svm_roc_auc = roc_auc_score(pca_ytest, svm_model.predict_proba(pca_Xtest), multi_class='ovr')
    
    roc_auc.append(svm_roc_auc)
    results.append(svm_accuracy)
    confusion_matrices.append(svm_conf_matrix)
''')

col9, col10 = st.columns(2)
with col9:
    st.write("We get these results:")
with col10:
    st.image("./plots/manymodels.png")
st.markdown("And obtained these confusion matrices:")
st.image('./plots/manycm.png')
st.markdown("Logistic Regression and most SVM models (except the one with the RBF kernel) faced challenges in learning from the principal components, while the other models performed reasonably well. However, none of them approached the performance of the baseline model, which learned from the original dataset without dimensionality reduction.")

st.markdown("## Tuning promising models 🔧🎛️")
st.markdown("### Tuning SVM with Grid Search 📔⚙️")
st.markdown("Lets tune the models which showed the most promise and see if we can beat our baseline accuracy score. I first started with using Grid Search and cross-validated the results of parameter combinations:")
st.code('''
from sklearn.model_selection import GridSearchCV

param_grid = {'C': np.linspace(0.1, 10, 20),
              'kernel': ['rbf'],
              'gamma': ['scale', 'auto']}
svm_model = SVC(probability=True)

print("Performing GridSearch...")
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(pca_Xtrain, pca_ytrain)
print("Done!")

# extract the grid search results
results = grid_search.cv_results_
mean_scores = results['mean_test_score']
std_scores = results['std_test_score']
param_C = results['param_C']
''')
st.markdown("A few images of the tuning process:") 
st.markdown("It was quite clear from the initial tune that gamma set to 'scale' produced better results across the board. For the second tune we tried finding the best value of C the regularization parameter, which came up to approximately 8.437.")
coltun1, coltun2 = st.columns(2)
with coltun1:
    st.image("./plots/gridsearch_results_initial_tune.png")
with coltun2:
    st.image("./plots/gridsearch_results_final_tune.png")
    
st.markdown("How did the tuned model perform on the test set? Unfortunately, it did not fare well. The accuracy reduced to 14.16%, which is significantly worse compared to the untuned model. The tuned hyperparameter configurations might have severely overfitted the train data.")    

st.markdown("### Tuning XGBoost with Random Search 🎲⚙️")
st.markdown("Let's try a different strategy with a different model! Here is a vanilla implementation of random search, with the benefit of more easily integrating a tqdm progress bar:")
st.code('''
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import random
import itertools

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
}
xgb_model = xgb.XGBClassifier(n_jobs=-1)
n_iter = 20

tqdm._instances.clear() # For avoiding duplicate bars in J-lab
pbar = tqdm(total=n_iter, desc='Random Search Progress')

best_results = []
for _ in range(n_iter):
    params = {param: random.choice(values) for param, values in param_grid.items()}
    xgb_model.set_params(**params)
    scores = cross_val_score(xgb_model, pca_Xtrain, pca_ytrain, cv=3, scoring='accuracy')
    mean_score = np.mean(scores)
    result = {'params': params, 'score': mean_score}
    
    best_results.append(result)
    best_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Keep only the top 5 results
    best_results = best_results[:5]
    pbar.update(1)
pbar.close()

print("Best 5 Results and Parameters:")
for i, result in enumerate(best_results):
    print(f"Rank {i+1}: Score = {result['score']}, Parameters = {result['params']}")
''')
p1, p2 = st.columns([2,3])
with p1:
    st.markdown("We obtain the following best params:")
with p2:
    st.image("./plots/bestparamsxgb.png")

st.markdown("#### Evaluating best params against train set")    
with open("./pickle_objects/xgbfinalresults.pkl", "rb") as file:
    xgbfinalresults = pickle.load(file) 

f1, f2 = st.columns([2,1])
with f1:
    st.markdown("How did the tuned models perform? Fortunately, we see slight improvements to the untuned models this time:")
with f2:
    st.dataframe(xgbfinalresults, hide_index=True)
    
st.image('./plots/xgb_finalcm.png')

st.markdown("## So what next? 🔚")

st.markdown("In this project, I employed Principal Component Analysis (PCA) to reduce the dimensionality of the dataset. Following dimensionality reduction, I fitted and tuned two models, expecting them to outperform a model trained on the original unreduced dataset. However, contrary to my expectations, both tuned models exhibited inferior performance compared to the model trained on the unreduced dataset. 😩 I strongly suspect that the limited nature of the data (only 50 samples per class for 23 classes) played a significant factor. There is still a lot for me to learn and improve on regarding PCA, dimensionality reduction, and dealing with temporal data like spectrometer readings.")

st.markdown('''
🙌🎉 Big thanks to you for reading my project post all the way to the end! 🙌🎉
Your time and attention are truly valued. I hope you found the project engaging and informative. ✨
''')


    
            


       


