import pandas as pd
import numpy as np
import seaborn as sns

from enum import Enum
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

from utils import plot_roc_curve
from utils import plot_confusion_matrix

sns.set()

def evaluate_model_ex(model, X_test, y_test, threshold = 0.5, c_matrix=True, r_curve=True):
    '''
    Function evaluates model and calculated various metrics. 
    INPUT:
    model - model to be evaluated
    X_test - Predictive part of the test dataset
    y_test - Expected values of the test dataset    
    c_matrix - flag if conf matrix should be plotted
    r_curve - flag is the ROC curve should be plotted

    OUTPUT:
    model - trained classifier
    metrics - dict with all evaliated metrics
    '''    
    metrics = dict()
    
    cmatrix = confusion_matrix(y_test, model.predict(X_test))
    
    TP = cmatrix[1][1] # should be 1, was 1
    FP = cmatrix[0][1] # should be 0, was 1

    TN = cmatrix[0][0] # should be 0, was 0
    FN = cmatrix[1][0] # should be 1, was 0

    if (TP+FP) > 0: # avoid division by Zero 
        precision = TP/(TP+FP)
    else:
        precision = 0
    
    if (TP+FN) > 0: # avoid division by Zero 
        recall = TP/(TP+FN)
    else:
        recall = 0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    
    metrics['accuracy'] = model.score(X_test, y_test)
    metrics['f1'] = f1_score(y_test, model.predict(X_test))
    metrics['auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # score
    print('Accuracy: {0:.4f}'.format(metrics['accuracy']))

    # F1 score    
    print('F1 Score: {0:.4f}'.format(metrics['f1']))

    # AUC-ROC score
    print('AUC-ROC Score: {0:.4f}'.format(metrics['auc']))

    print('-----------------------------------------')
    
    # TPR, FPR
    print('Precision: {0:.4f}'.format(metrics['precision']))
    print('Recall: {0:.4f}'.format(metrics['recall']))

    print('=========================================')
    
    # Confusion Matrix
    if c_matrix == True:
        plot_confusion_matrix(model, X_test, y_test)

    # ROC curve
    if r_curve == True:
        plot_roc_curve(model, X_test, y_test)
        
    return model, metrics
  
def evaluate_model(model, X_test, y_test, threshold=.5, c_matrix=True, r_curve=True):
    '''
    Function evaluates model and calculated various metrics. 
    INPUT:
    model - model to be evaluated
    X_test - Predictive part of the test dataset
    y_test - Expected values of the test dataset    
    c_matrix - flag if conf matrix should be plotted
    r_curve - flag is the ROC curve should be plotted

    OUTPUT:
    model - trained classifier
    metrics - dict with all evaliated metrics
    '''    
    metrics = dict()
    
    preds = np.where(model.predict_proba(X_test)[:,1] > threshold, 1, 0)
    
    metrics['accuracy'] = accuracy_score(y_test, preds)
    metrics['f1'] = f1_score(y_test, preds)
    metrics['auc'] = roc_auc_score(y_test, preds)

    metrics['precision'] = precision_score(y_test, preds)
    metrics['recall'] = recall_score(y_test, preds)
    
    # score
    print('Accuracy: {0:.4f}'.format(metrics['accuracy']))

    # F1 score    
    print('F1 Score: {0:.4f}'.format(metrics['f1']))

    # AUC-ROC score
    print('AUC-ROC Score: {0:.4f}'.format(metrics['auc']))

    print('-----------------------------------------')
    
    # TPR, FPR
    print('Precision: {0:.4f}'.format(metrics['precision']))
    print('Recall: {0:.4f}'.format(metrics['recall']))

    print('=========================================')
    
    # Confusion Matrix
    if c_matrix == True:
        plot_confusion_matrix(model, X_test, y_test)

    # ROC curve
    if r_curve == True:
        plot_roc_curve(model, X_test, y_test)
        
    return model, metrics


def init_metrics_file():
    '''
    Function initializing the CSV file for metrics
    INPUT: no
   
    OUTPUT: no
    '''     
        
    idx = pd.MultiIndex.from_product([['LogReg', 'LGBM', 'KNC'],
                                      ['accuracy', 'f1-score', 'auc-roc', 'precision', 'recall']],
                                     names=['Classifier', 'Metric'])
    col = ['1_unbalanced', '2_weighted', '3_balanced', '4_with_distance', '5_with_angle', '6_with_player_ids', '6_with_player_ids_weighted', '7_with_player_stats', '7_with_player_stats_weighted', '8_with_player_salary', '8_with_player_salary_weighted', '9_short_dist','9_short_dist_weighted', '10_long_dist', '10_long_dist_weighted' ]

    df = pd.DataFrame('-', idx, col)
    df.to_csv('data/results.csv')

def save_metrics(metric_id, metrics_lg=None, metrics_lgbm=None, metrics_knc=None):
    '''
    Function saving metrics (accuracy, f1, auc) of one or more models to predefined file.
    The function can be called for all classifier (LogReg, LGBM, KNC) at one, or separately.
    INPUT:
    metric_id - column to store the data to (id of the dataset)
    metric_lg - accuracy, f1, auc calculated from a logistic regression classifier
    metric_lgbm - accuracy, f1, auc calculated from a LGBM classifier
    metric_knc - accuracy, f1, auc calculated from a KNeighbors classifier
    OUTPUT: no
    '''     
    
    df = pd.read_csv('data/results.csv')
    df.set_index(['Classifier', 'Metric'], inplace=True)
    
    if metrics_lg is not None:
        df.at[('LogReg', 'accuracy'), metric_id] = metrics_lg['accuracy']
        df.at[('LogReg', 'f1-score'), metric_id] = metrics_lg['f1']
        df.at[('LogReg', 'auc-roc'), metric_id] = metrics_lg['auc']
        df.at[('LogReg', 'precision'), metric_id] = metrics_lg['precision']
        df.at[('LogReg', 'recall'), metric_id] = metrics_lg['recall']

    if metrics_lgbm is not None:    
        df.at[('LGBM', 'accuracy'), metric_id] = metrics_lgbm['accuracy']
        df.at[('LGBM', 'f1-score'), metric_id] = metrics_lgbm['f1']
        df.at[('LGBM', 'auc-roc'), metric_id] = metrics_lgbm['auc']
        df.at[('LGBM', 'precision'), metric_id] = metrics_lgbm['precision']
        df.at[('LGBM', 'recall'), metric_id] = metrics_lgbm['recall']

    if metrics_knc is not None:
        df.at[('KNC', 'accuracy'), metric_id] = metrics_knc['accuracy']
        df.at[('KNC', 'f1-score'), metric_id] = metrics_knc['f1']
        df.at[('KNC', 'auc-roc'), metric_id] = metrics_knc['auc']    
        df.at[('KNC', 'precision'), metric_id] = metrics_knc['precision']
        df.at[('KNC', 'recall'), metric_id] = metrics_knc['recall']
    
       
    df.to_csv('data/results.csv')
    
def plot_metrics():
    '''
    Function opens the file with save metrics, stores it in a DataFrame and return the DF for next processing
    INPUT: no
    OUTPUT: DataFrame with the metrics structure
    '''      
    
    df = pd.read_csv('data/results.csv')
    df.set_index(['Classifier', 'Metric'], inplace=True)
    
    print()
    
    return df