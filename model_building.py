# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:32:44 2020

@author: Anastasia
"""
import pandas as pd 
import numpy as np 
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# import file 

df = pd.read_csv('cleaned_data.csv', index_col=0)

# cleaning from the EDA

df['No-show'] = df['No-show'].apply(lambda x: 1 if x == 'No' else 0)
df = df[df['Age'] > 0]
df = df[df['Sched_to_App_Time'] >= 0]
df['Handcap'] = df['Handcap'].apply(lambda x: 0 if x == 0 else 1)

# choose relevant columns for model building 

df_nums = df[['Age', 'Sched_to_App_Time']]

# get dummy variables 

df_dum = pd.get_dummies(df[['Gender', 'Neighbourhood', 'Scholarship', 'Hipertension','Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show']])

df_model = pd.concat([df_nums, df_dum], axis=1)

# train test split 

X = df_model.drop('No-show', axis=1)
y = df_model['No-show'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# logistic regression

lr = LogisticRegression(solver='lbfgs', max_iter=500)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test) 

print(accuracy_score(y_test, y_pred_lr))

# SGD

scaler = StandardScaler()
sgd = SGDClassifier()

pipeline = Pipeline([('scaler', scaler), ('sgd', sgd)])

pipeline.fit(X_train,y_train)

y_pred_pipe = pipeline.predict(X_test) 

print(accuracy_score(y_test, y_pred_pipe))

# K-Nearest Neighbours 

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test) 

print(accuracy_score(y_test, y_pred_knn))

# Random Forest

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test) 

print(accuracy_score(y_test, y_pred_rf))

# tune models using GridSearchCV

# KNN Tuning

leaf_size = list(range(1,50,10))
n_neighbors = list(range(1,30,5))
p=[1,2]

parameters_knn = {'leaf_size':leaf_size, 'n_neighbors':n_neighbors, 'p':p}

gs_knn = GridSearchCV(knn,parameters_knn,scoring='accuracy',cv=3)
gs_knn.fit(X_train,y_train)

print(gs_knn.best_score_)
print(gs_knn.best_estimator_)

gsknn_pred = gs_knn.best_estimator_.predict(X_test)

# SGD Tuning

parameters_sgd = {"sgd__n_iter_no_change": [1, 5, 10], "sgd__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], "sgd__penalty": ["none", "l1", "l2"]}
 
gs_sgd = GridSearchCV(pipeline,parameters_sgd,scoring='accuracy',cv=3)
gs_sgd.fit(X_train,y_train)

print(gs_sgd.best_score_)
print(gs_sgd.best_estimator_)

gssgd_pred = gs_sgd.best_estimator_.predict(X_test)

cmat_gsknn = confusion_matrix(y_test, gsknn_pred)
cmat_gssgd = confusion_matrix(y_test, gssgd_pred)

def heatmap_labels(x):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                x.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     x.flatten()/np.sum(x)]
    labels = [f"{v1}\n{v2}\n{v3}"for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    return sns.heatmap(x, annot=labels, fmt='', cmap='Blues')

heatmap_labels(cmat_gsknn)
heatmap_labels(cmat_gssgd)

