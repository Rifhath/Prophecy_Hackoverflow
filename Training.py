import pandas as pd
import numpy as np
from numpy import loadtxt
import xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from hyperopt import hp
import random

dataset = pd.read_csv(r'C:\Users\RIFHATH ASLAM\OneDrive\Desktop\Sentimental analysis.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat = dataset.select_dtypes(include='O').keys()
cat = list(cat)
for i in cat:
  dataset[i] = le.fit_transform(dataset[i])
for i in dataset.columns:
  dataset[i].fillna(int(dataset[i].mean()), inplace=True)
dataset.to_csv('file2.csv', header=False, index=False)
dataset = loadtxt('file2.csv', delimiter=",")
X=dataset[:,0:-1]
Y=dataset[:,-1]
kfold = KFold(n_splits=10)

kfoldn = KFold(n_splits=250)
abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
parameters = {'base_estimator__max_depth':[i for i in range(2,11,2)], 'base_estimator__min_samples_leaf':[5,10], "base_estimator__criterion" : ["gini", "entropy"], "base_estimator__splitter" :   ["best", "random"], 'n_estimators':[10,50,250,1000], 'learning_rate':[0.01,0.1]}
clf2 = GridSearchCV(abc, parameters,verbose=3,scoring='f1',n_jobs=-1)
clf2.fit(X, Y)
print(clf2.best_params_)
print(clf2.score(X, Y))