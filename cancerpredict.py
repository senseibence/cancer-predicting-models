import pandas
import numpy
import xgboost as xgb
from sklearn.model_selection import train_test_split 
from sklearn import tree

''' Classifier will be for no/yes (0/1) when I get medical data'''
from sklearn.tree import DecisionTreeClassifier 

''' This is for classifiers only '''
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

encode = LabelEncoder()

df = pandas.read_csv('testdata.csv')
'''df = pandas.read_csv('categoricaloutput.csv')'''

inputs = ['Open','High','Low']
X = df[inputs]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

'''dtree = DecisionTreeClassifier()''' 
dtree = DecisionTreeRegressor()
dtree = dtree.fit(X_train,y_train) 
'''regularDTprediction = dtree.predict(X_test)'''

xgbtree = xgb.XGBRegressor()
xgbtree = xgbtree.fit(X_train,y_train)
'''XGBoostDTprediction = xgbtree.predict(X_test)'''

''' R^2 is 0.9999... for both models '''
print("Accuracy of Model:",dtree.score(X_test, y_test))
print("Accuracy of Model:",xgbtree.score(X_test, y_test))

''' 
Next Goals | Primary :
Try out sklearn random forest algorithm, tensorflow neural network, 
the categorial data, and adding parameters to XGBoost models

Next Goals | Secondary :
Try logistic regression, K-neighbors, guassianNB, support vector machines, etc (all in bookmarked links)
'''