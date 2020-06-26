# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:58:40 2020
@author: Ronald Tumuhairwe
This project compares the Isolation Forest and LocalOutlierFactor algorithms (anomaly detection algorithms) 
for credit card detection. In this project the Isolation Forest algorithm provides better precision and recall for fraud cases
compared to the LocalOutlier Factors 

The dataset used for the project can be found https://www.kaggle.com/mlg-ulb/creditcardfraud/data 

For compution resources purposes, we will use only 20% of the data for this project however you can 
use the entire data 
"""
#Import the necessary libaries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset from the CSV file
data = pd.read_csv('creditcard.csv')

#We use these methods to learn more about the dataset composition
print(data.columns)
print(data.describe())
print(data.shape)

#We will only sample a few of the transcations for compution purposes
data = data.sample(frac=0.2, random_state = 1)
print(data.shape)

#plot histograms of each parameter to visualize the data, this enables to you understand more about the data
#composition of each column such as the mean
#data.hist(figsize = (20,20))
#plt.show() 

#determine the number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)

print('Fraud case:{}'.format(len(fraud)))
print('valid cases:{}'.format(len(valid)))

#check for correlation between our variables and which features are important (significantly dependent/independent)
#using a correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize =(12,9))

#heatmap to visualize the correlation matrix of our data/ In this case, the Class column has significant dependence on all other 
#columns because its state shows the case is fraud/ true based on data in othre columns 
sns.heatmap(corrmat,vmax= .8, square=True)

#Get all columns from the Dataframe
columns = data.columns.tolist()

#filter the columns to remove data we do not want
#we remove the 'Class' column since this is unsupervised learning and we want to predict the 'Class' value
# if the case is legit or fraud other than feeding it directly (which defeats our purpose of the project)
columns = [c for c in columns if c not in ["Class"]]

#Store the variable we will be predicting on
target = "Class"

X = data[columns]
Y = data[target]

#Print the shapes of X and Y for clarity
print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state = 1

#define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples = len(X),
                                        contamination = outlier_fraction,
                                        random_state=state),
    "Local_Outlier Factor": LocalOutlierFactor(
        n_neighbors = 20, 
        contamination = outlier_fraction, novelty = True)
    
    }

#fit our model
n_outliers = len(fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    #Reshape the prediction values to 0 for valid, 1 for fraud
        y_pred[y_pred ==1] = 0
        y_pred[y_pred == -1] = 1
        
        n_errors = (y_pred != Y).sum()
        
    #Run classification metrics 
        print('{}: {}'.format(clf_name, n_errors))
        print(accuracy_score(Y,y_pred))
        print(classification_report(Y,y_pred)) 
        
    
    
    




