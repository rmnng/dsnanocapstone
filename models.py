import pandas as pd
import numpy as np
import math
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

from metrics import evaluate_model

from matplotlib import pyplot as plt

sns.set()

def run_lightgbm_classifier(df, target='goal', unbalanced=False, features=None, test_size = 0.3, c_matrix=True, r_curve=True):

    '''
    Function executed the LGBM classification and calls the results evaluation. 
    INPUT:
    df - data frame with data used for the LGBM classifier
    features - list of predictive features used in the model
    target - target to be predicted
    unbalanced - flag if to balance the dataset
    test_size - portion of the test part of the dataset
    c_matrix - flag indicating plotting of the confusion matrix
    r_curve - flag indicating plotting of the ROC curve

    
    OUTPUT:
    model - trained LGBM classifier model
    metrics - dict with acc, f1, auc
    ''' 
    if features == None:
        features = df.select_dtypes(exclude='object').columns.to_list()
        features.remove(target)

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size, random_state=42)

    # create model
    model =  LGBMClassifier(random_state=42, is_unbalance=unbalanced)

    # fit model
    model.fit(X_train,y_train)
        
    print('========== LightGBM Classifier ==========')

    return evaluate_model(model, X_test, y_test, c_matrix=c_matrix, r_curve=r_curve)


def run_logistic_regression(df, target='goal', features=None, test_size = 0.3, balance_weights = False, max_iter=10000, c_matrix=True, r_curve=True):
    '''
    Function executed the logistic regression and calls the results evaluation. 
    INPUT:
    df - data frame with data used for the LGBM classifier
    features - list of predictive features used in the model
    target - target to be predicted
    max_iter - max interations of the model during training
    balance_weights - flag if weights for target values should be balanced
    test_size - portion of the test part of the dataset
    c_matrix - flag indicating plotting of the confusion matrix
    r_curve - flag indicating plotting of the ROC curve
    
    OUTPUT:
    model - trained logistic regression model
    metrics - dict with acc, f1, auc
    '''

    # select features as
    if features == None:
        features = df.select_dtypes(exclude='object').columns.to_list()
        features.remove(target)
        
    # balance weights
    if balance_weights == True:
        weights = { 0:df[df[target] == 1].shape[0]/df.shape[0], 
                    1:df[df[target] == 0].shape[0]/df.shape[0]}
    else:
        weights = { 0:1.0, 
                    1:1.0}

        
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size, random_state=42)

    # create model
    model = LogisticRegression(max_iter=max_iter, class_weight=weights);

    # fit model
    model.fit(X_train,y_train)

    print('========== Logistic Regression ==========')

    return evaluate_model(model, X_test, y_test, c_matrix=c_matrix, r_curve=r_curve)

def run_kneighbors_classifier(df, target='goal', features=None, test_size = 0.3, n_neighbors=100, c_matrix=True, r_curve=True):
    '''
    Function executed the KNeighbors Classifier and calls the results evaluation. 
    INPUT:
    df - data frame with data used for the LGBM classifier
    features - list of predictive features used in the model
    target - target to be predicted
    test_size - portion of the test part of the dataset
    n_neighbors - hyper parameter of the model
    c_matrix - flag indicating plotting of the confusion matrix
    r_curve - flag indicating plotting of the ROC curve 

    OUTPUT:
    model - trained classifier
    metrics - dict with acc, f1, auc
    '''
    if features == None:
        features = df.select_dtypes(exclude='object').columns.to_list()
        features.remove(target)

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size, random_state=42)

    # create model
    model = KNeighborsClassifier(n_neighbors = n_neighbors)

    # fit model
    model.fit(X_train,y_train)

    print('========= KNeighbors Classifier =========')
  
    return evaluate_model(model, X_test, y_test, c_matrix=c_matrix, r_curve=r_curve)

