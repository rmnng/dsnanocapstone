import pandas as pd
import numpy as np
import math
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

from lightgbm import LGBMClassifier

from matplotlib import pyplot as plt

sns.set()

def balance_binary_target(df, target):
    '''
    Function balancing dataset to have 50/50 distribution of binary target values (0/1)
    INPUT:
    df - data frame to be balanced
    target - name of the target column based on which the dataset should be balanced
    
    OUTPUT: no
    '''     
    size = min(df[df[target]==0].shape[0], df[df[target]==1].shape[0])
    
    goals = df[df[target]==1].sample(size, replace=False, random_state=10)
    no_goals = df[df[target]==0].sample(size, replace=False, random_state=10)
    
    return pd.concat([goals, no_goals])

def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
    '''
    
    for col in cat_cols:
        try:
            df = pd.concat([df.drop(columns=col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            pass
    
    return df.copy()


def plot_roc_curve(model, X_test, y_test):
    '''
    Function plotting ROC curve
    INPUT:
    model - model to plot the ROC curve for
    X_test - Predictive part of the test dataset
    y_test - Expected values of the test dataset
    
    OUTPUT: no
    ''' 
    prob_preds = model.predict_proba(X_test)[:,1]
    fp_rate, tp_rate, _ = roc_curve(y_test,  prob_preds)

    # plot ROC curve
    plt.figure(figsize=(5,5))
    plt.axline([0, 0], [1, 1], color='red', linestyle='dashed',label='random guess')
    plt.plot(fp_rate, tp_rate, label='roc curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def plot_correlation_matrix(df_data, col=None):
    '''
    Function plotting correlation matrix of a data frame
    INPUT:
    df_data - data frame to plot the correlation matrix for
    col - column to make the correlation for
    
    OUTPUT: no
    '''    
    corr = df_data.corr()
    cmap = sns.diverging_palette(250, 20, as_cmap=True)
    
    if col==None:
        sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, linewidths=1, cbar_kws={"shrink": .5});
    else:
        plt.title(col)
        sns.heatmap(corr[[col]], cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, linewidths=1, cbar_kws={"shrink": .5});
        

def calculate_confusion_matrix(y_test, y_pred, plot=False):
    '''
    Function plotting confusion matrix as a heatmap.
    INPUT:
    y_test - 1d vector with expected values
    y_pred - 1d vector with predicted values
    
    OUTPUT: no
    '''    
    
    cmatrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if plot == True:
        sns.heatmap(cmatrix, annot=True,  fmt=".0f", linewidths=1, square = True, cmap='Reds', linecolor='black', cbar=False);
        plt.ylabel('Real');
        plt.xlabel('Predicted');
        plt.title(f'F1 Score: {f1:.4f}')
    
    
def plot_confusion_matrix(model, y_test, preds):    
    '''
    Function plotting confusion matrix for a trained model
    INPUT:
    model - model to plot the confusion matrix for
    X_test - Predictive part of the test dataset
    y_test - Expected values of the test dataset    
    OUTPUT: no
    ''' 

    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_ )
    disp.plot()
    plt.show()
    

    
