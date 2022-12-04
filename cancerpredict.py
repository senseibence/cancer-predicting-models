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

import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.python.keras.models import Model
from keras.utils import plot_model

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
    predicted = xgbtree.predict(X_test)
    
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

    print(metrics.classification_report(y_test, predicted))

    Accuracy = metrics.accuracy_score(y_test, predicted)
    Precision = metrics.precision_score(y_test, predicted)
    Sensitivity_recall = metrics.recall_score(y_test, predicted)
    Specificity = metrics.recall_score(y_test, predicted, pos_label=0)
    F1_score = metrics.f1_score(y_test, predicted)

    #metrics:
    print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})
    
def neural_network():
    df = pandas.read_csv('largecategoricaltestdata.csv', sep=';')
    
    categoricalConversions = {'yes': 1, 'no': 0}
    df['y'] = df['y'].map(categoricalConversions)
    y = df['y']

    df = pandas.get_dummies(df,df.columns[df.dtypes == 'object'])
    X = df.drop('y', axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    input_layer = Input(shape=(X.shape[1],)) ;'63'
    dl1 = Dense(32, activation='relu')(input_layer)
    dl2 = Dense(16, activation='relu')(dl1)
    output_layer = Dense(1, activation='sigmoid')(dl2)

    model = Model(inputs = input_layer, outputs=output_layer)
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    History = model.fit(X_train, y_train, epochs=75, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test,  y_test) 
    print('\n'+'Accuracy:', accuracy)
    print('Loss:', loss)

    'predicted = model.predict(X_test)'
    'print(History.history.keys())'

    # summarize history for accuracy
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('accuracy vs epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('loss vs epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # combined
    pandas.DataFrame(History.history).plot(figsize=(7,5))
    plt.show()

    'print(metrics.roc_auc_score(y_test, predicted))'

'Function Calls:' 
'regressor_models()'
'classifier_models()'
'neural_network()'

'''
Goals:
1) XGBoost parameters
2) Penalize neural network for favoring class 0 ("No")
3) Start making website; hope PLCO data request goes through
'''