import pandas
import numpy
import xgboost as xgb
'''from sklearn.model_selection import train_test_split '''
from sklearn import tree
'''from sklearn.tree import DecisionTreeClassifier // Classifier will be for no/yes (0/1) when I get medical data'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

encode = LabelEncoder()

df = pandas.read_csv('allvariables&p500.csv')
'''df = pandas.read_csv('old.csv')'''

'''
print(df)
print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
'''

inputs = ['Open','High','Low']

X = df[inputs]
y = df['Close']

'''dtree = DecisionTreeClassifier()''' 
dtree = DecisionTreeRegressor()
dtree = dtree.fit(X.values, y)

tree.plot_tree(dtree, feature_names=inputs)

''' 3657.1,	3757.89, 3647.42, 3752.75 '''
print(dtree.predict([[3830, 4120, 3663.92,]]))


'''
train_X = X[:80]
train_y = y[:80]

test_X = X[80:]
test_y = y[80:]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

train_y = encode.fit_transform(train_y)
test_y = encode.fit_transform(test_y)

dTree_clf = DecisionTreeClassifier()
xgb_classifier = xgb.XGBClassifier()

dTree_clf.fit(train_X,train_y)
xgb_classifier.fit(train_X,train_y)

pred2_y = dTree_clf.predict(test_X)
predictions = xgb_classifier.predict(test_X)

print("Accuracy of Model::",accuracy_score(test_y,pred2_y))
print("Accuracy of Model::",accuracy_score(test_y,predictions))
'''
