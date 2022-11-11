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
    X = df.drop('y', axis = 1)
    X = scale.fit_transform(X)

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
    
    '''
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    '''

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X_train,y_train)
    
    xgbtree = xgb.XGBClassifier()
    xgbtree = xgbtree.fit(X_train,y_train)
    'predicted = xgbtree.predict(X_test)'
    
    randomForest = RandomForestClassifier()
    randomForest = randomForest.fit(X_train,y_train)

    logisticModel = LogisticRegression()
    logisticModel = logisticModel.fit(X_train,y_train)

    kneighborsModel = KNeighborsClassifier()
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

    '''
    Accuracy = metrics.accuracy_score(y_test, predicted)
    Precision = metrics.precision_score(y_test, predicted)
    Sensitivity_recall = metrics.recall_score(y_test, predicted)
    Specificity = metrics.recall_score(y_test, predicted, pos_label=0)
    F1_score = metrics.f1_score(y_test, predicted)

    #metrics:
    print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})
    '''

'TensorFlow'
import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.python.keras.models import Model
from keras.utils import plot_model

def neural_network():
    df = pandas.read_csv('largecategoricaltestdata.csv', sep=';')
    
    categoricalConversions = {'yes': 1, 'no': 0}
    df['y'] = df['y'].map(categoricalConversions)
    y = df['y']

    df = pandas.get_dummies(df,df.columns[df.dtypes == 'object'])
    X = df.drop('y', axis = 1)
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

    ip_layer = Input(shape=(X.shape[1],))
    dl1 = Dense(100, activation='relu')(ip_layer)
    dl2 = Dense(50, activation='relu')(dl1)
    dl3 = Dense(25, activation='relu')(dl2)
    dl4 = Dense(10, activation='relu')(dl3)
    output = Dense(1)(dl4)

    model = Model(inputs = ip_layer, outputs=output)
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    history = model.fit(X_train, y_train, batch_size=5, epochs=10, verbose=1, validation_split=0.2)

    '''plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()'''

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=1) 
    print('Accuracy:', test_acc)
    print('Loss:', test_loss)

    '''y_pred = model.predict(X_test)
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))'''

    '''print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))'''

'Function Calls' 
'regressor_models()'
'classifier_models()'
neural_network()

'''
Goals:
1) Tensorflow neural network
3) XGBoost parameters
'''