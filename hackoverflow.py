import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from hyperopt import hp
import random
from flask import Flask, render_template,request
import pickle

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
#{'base_estimator__criterion': 'entropy', 'base_estimator__max_depth': 6, 'base_estimator__min_samples_leaf': 5, 'base_estimator__splitter': 'best', 'learning_rate': 0.01, 'n_estimators': 50}
abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion= "entropy",max_depth= 6, min_samples_leaf= 5, splitter= "random"), learning_rate= 0.01, n_estimators= 50)
abc.fit(X, Y)
pickle.dump(abc, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,0,0]]))

app=Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features = np.array(final_features)
    final_features = final_features[0, 0:-1]
    prediction = model.predict([final_features])
    output = round(prediction[0], 2)
    if output == 0.0:
        finout = "Depressed"
    else:
        finout = "Healthy"
    return render_template('index.html', prediction_text="You're :{}".format(finout))

if __name__ == "__main__":
    app.run(debug=True)
