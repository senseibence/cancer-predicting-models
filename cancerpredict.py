from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

import pandas
import numpy
import matplotlib.pyplot as plt
import xgboost as xgb

def regressor_models():
    df = pandas.read_csv('largenumericaltestdata.csv')
    '''df = pandas.read_csv('smallnumericaltestdata.csv')'''
    
    '''inputs = ['Day']'''
    inputs = ['Open','High','Low']
    X = df[inputs]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
 
    dtree = DecisionTreeRegressor()
    dtree = dtree.fit(X_train,y_train) 

    xgbtree = xgb.XGBRegressor()
    xgbtree = xgbtree.fit(X_train,y_train)

    randomForest = RandomForestRegressor(random_state = 42)
    randomForest = randomForest.fit(X_train,y_train)

    kneighborsModel = KNeighborsRegressor()
    kneighborsModel = kneighborsModel.fit(X_train,y_train)

    print("Accuracy of DT Regressor Model:",dtree.score(X_test, y_test))
    print("Accuracy of XG Regressor Model:",xgbtree.score(X_test, y_test))
    print("Accuracy of RF Regressor Model:",randomForest.score(X_test, y_test))
    print("Accuracy of KN Regressor Model:",kneighborsModel.score(X_test, y_test))
    
def classifier_models():
    df = pandas.read_csv('largecategoricaltestdata.csv', sep=';')
    categoricalConversions = {'yes': 1, 'no': 0}
    df['y'] = df['y'].map(categoricalConversions)
    y = df['y']

    df = pandas.get_dummies(df,df.columns[df.dtypes == 'object'])
    X = df.iloc[:, 0:63]
    '''X = scale.fit_transform(X)'''

    '''
    df = pandas.read_csv('smallcategoricaltestdata.csv')

    categoricalConversions = {'UK': 0, 'USA': 1, 'N': 2}
    df['Nationality'] = df['Nationality'].map(categoricalConversions)
    categoricalConversions = {'YES': 1, 'NO': 0}
    df['Go'] = df['Go'].map(categoricalConversions)

    inputs = ['Age','Experience','Rank','Nationality']
    X = df[inputs]
    y = df['Go']
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X_train,y_train)
    
    xgbtree = xgb.XGBClassifier(n_estimators = 2, max_depth = 3)
    xgbtree = xgbtree.fit(X_train,y_train)
    ''' xgbPrediction = xgbtree.predict(X_test)'''
    
    randomForest = RandomForestClassifier(random_state = 42)
    randomForest = randomForest.fit(X_train,y_train)

    logisticModel = LogisticRegression()
    logisticModel = logisticModel.fit(X_train,y_train)

    kneighborsModel = KNeighborsClassifier(n_neighbors = 6)
    kneighborsModel = kneighborsModel.fit(X_train,y_train)
 
    naiveBayesModel = GaussianNB()
    naiveBayesModel = naiveBayesModel.fit(X_train,y_train)

    linDiscrimModel = LinearDiscriminantAnalysis()
    linDiscrimModel = linDiscrimModel.fit(X_train,y_train)
    
    print("Accuracy of DT Classifier Model:",dtree.score(X_test, y_test))
    print("Accuracy of XG Classifier Model:",xgbtree.score(X_test, y_test))
    print("Accuracy of RF Classifier Model:",randomForest.score(X_test, y_test))
    print("Accuracy of LR Classifier Model:",logisticModel.score(X_test, y_test))
    print("Accuracy of KN Classifier Model:",kneighborsModel.score(X_test, y_test))
    print("Accuracy of NB Classifier Model:",naiveBayesModel.score(X_test, y_test))
    print("Accuracy of LD Classifier Model:",linDiscrimModel.score(X_test, y_test))

    '''print(metrics.classification_report(y_test,xgbPrediction))'''

''' Function Calls '''
'''classifier_models()'''
'''regressor_models()'''

''' 
Goals:
1) Understand why accuracy is 100%
2) Tensorflow neural network
3) XGBoost parameters
'''