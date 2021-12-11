import pandas as pd
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from lightgbm import LGBMClassifier

from matplotlib import pyplot as plt

sns.set()

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


def plot_correlation_matrix(df_data, col=None):
    '''
    Function plotting correlation matrix of a data frame
    INPUT:
    df_data - data frame to plot the correlation matrix for
    
    OUTPUT: no
    '''    
    corr = df_data.corr()
    plt.subplots(figsize=(22, 18))
    cmap = sns.diverging_palette(250, 20, as_cmap=True)
    
    if col==None:
        sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, square=True, linewidths=1, cbar_kws={"shrink": .5});
    else:
        plt.title(coll)
        sns.heatmap(corr[[col]], cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, linewidths=1, cbar_kws={"shrink": .5});
        



def run_model(model, df_data, lst_features, target):
    '''
    INPUT:
    model - model to be used
    df_data - data frame with data used for the training and scoring of the model
    lst_features - list of predictive features used in the model
    target - target to be predicted
    
    OUTPUT:
     X_test - data used for test
     y_test - targets used for test
    '''    
    
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df_data[lst_features], df_data[target], test_size=0.3, random_state=42)

    # start trainig
    start = time.time()
    model.fit(X_train,y_train)
    duration = time.time() - start
    print(f'Training ran in {duration:.5f} seconds')    

    # use model score
    score = model.score(X_test, y_test)    
    print(f'Score: {score}')
    
    #calculate f1 score
    f1 = f1_score(y_test, model.predict(X_test))
    print(f'F1 Score: {f1:.2f}')
    
    return model, X_test, y_test
    

def run_logistic_regression(df_data, lst_features, target, max_iter = 10000):
    '''
    INPUT:
    df_data - data frame with data used for the logistic regression
    lst_features - list of predictive features used in the model
    target - target to be predicted
    
    OUTPUT:
    model - trained logistic regression model
    X_test - data used for test
    y_test - targets used for test
    '''    

    return run_model(LogisticRegression(max_iter=max_iter), df_data, lst_features, target);

def run_lightgbm_classifier(df_data, lst_features, target):
    '''
    INPUT:
    df_data - data frame with data used for the LightGMB classifier
    lst_features - list of predictive features used in the model
    target - target to be predicted
    
    OUTPUT:
    model - trained logistic regression model
    X_test - data used for test
    y_test - targets used for test    
    '''    
    
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df_data[lst_features], df_data[target], test_size=0.3, random_state=42)

    model = LGBMClassifier(n_estimators=100,
                         num_leaves=64,
                         max_depth=3,
                         learning_rate=0.1,
                         random_state=1000,
                         n_jobs=-1);
    
    return run_model(model, df_data, lst_features, target)

def test_logistic_regression(df_data, lst_features, target, plot_cmatrix=False, max_iter = 10000):
    '''
    INPUT:
    df_data - data frame with data used for the logistic regression
    lst_features - list of predictive features used in the model
    target - target to be predicted
    
    OUTPUT:
    model - trained logistic regression model
    X_test - data used for test
    y_test - targets used for test    
    '''    

    model = LogisticRegression(penalty='l2', tol=0.0001, 
                           C=1, fit_intercept=True, intercept_scaling=2.0, 
                           class_weight=None, random_state=42, max_iter=max_iter);
    
    return run_model(model, df_data, lst_features, target)

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
    print(f'F1 Score: {f1:.2f}')
    
    if plot == True:
        sns.heatmap(cmatrix, annot=True,  fmt=".0f", linewidths=1, square = True, cmap='Reds', linecolor='black');
        plt.ylabel('Real');
        plt.xlabel('Predicted');
    
    return f1, cmatrix

    