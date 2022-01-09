#########Notes############
"""
Predict alpha with the relevant features by applying SVM

1. relevant features: technical features & sentiment volatility, sentiment momentum, normalized average sentiment as a sentimental features
2. Use SVM:The  3  fold  cross  validation / RBF-kernel
"""
########Libraries########
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

########Functions########
def add_alpha_boolean(df):
    for i, row in df.iterrows():
        if row['next day return'] >= 0:
            sig = int(1)
        else:
            sig= int(0)
        df.at[i, 'positive alpha'] = sig
    return df

def baseline(feature_list, feature_df, figure_3_df):
    print("start new metric with", feature_list)
    print(list(feature_df.iloc[:,feature_list].columns.values))
    X_train, X_test, y_train, y_test = train_test_split(feature_df.iloc[:,feature_list], feature_df.iloc[:,14], test_size=0.3,random_state=42)

    #GridSearch
    clf = LogisticRegression()
    param_grid = {'penalty': ['l1'],
               'C':[0.001,.009,0.01,.09,1,5,10,25],
               'solver':['liblinear']}

    grid_clf_acc = GridSearchCV(clf, param_grid = param_grid,scoring = 'accuracy')
    grid_clf_acc.fit(X_train, y_train)
    grid_predictions = grid_clf_acc.predict(X_test)
    print(grid_clf_acc.best_params_)

    #Predict values based on new parameters
    y_pred_acc = grid_clf_acc.predict(X_test)
    #add metrcis to df
    new_row = {'Model':"Baseline",'Features':list(feature_df.iloc[:,feature_list].columns.values),'Accuracy Score':accuracy_score(y_test,y_pred_acc), 'Precision Score':precision_score(y_test,y_pred_acc), 'Recall Score':recall_score(y_test,y_pred_acc), 'F1 Score ':f1_score(y_test,y_pred_acc)}
    figure_3_df= figure_3_df.append(new_row, ignore_index=True)

    return figure_3_df
def sentiment_logreg(feature_list, feature_df, figure_3_df):
    print("start new metric with", feature_list)
    print(list(feature_df.iloc[:,feature_list].columns.values))
    X_train, X_test, y_train, y_test = train_test_split(feature_df.iloc[:,feature_list], feature_df.iloc[:,14], test_size=0.3,random_state=42)

    #GridSearch
    clf = LogisticRegression()
    param_grid = {'penalty': ['l1'],
               'C':[0.001,.009,0.01,.09,1,5,10,25],
               'solver':['liblinear']}

    grid_clf_acc = GridSearchCV(clf, param_grid = param_grid,scoring = 'accuracy')
    grid_clf_acc.fit(X_train, y_train)
    grid_predictions = grid_clf_acc.predict(X_test)
    print(grid_clf_acc.best_params_)

    #Predict values based on new parameters
    y_pred_acc = grid_clf_acc.predict(X_test)
    #add metrcis to df
    new_row = {'Model':"Sentiment + Logistic Regression",'Features':list(feature_df.iloc[:,feature_list].columns.values),'Accuracy Score':accuracy_score(y_test,y_pred_acc), 'Precision Score':precision_score(y_test,y_pred_acc), 'Recall Score':recall_score(y_test,y_pred_acc), 'F1 Score ':f1_score(y_test,y_pred_acc)}
    figure_3_df= figure_3_df.append(new_row, ignore_index=True)

    return figure_3_df
def sentiment_svm(feature_list, feature_df, figure_3_df):
    print("start new metric with", feature_list)
    print(list(feature_df.iloc[:,feature_list].columns.values))
    X_train, X_test, y_train, y_test = train_test_split(feature_df.iloc[:,feature_list], feature_df.iloc[:,14], test_size=0.5,random_state=42)

    #GridSearch
    svc = svm.SVC()
    param_grid = {'C': [0.35,0.4,0.45],
                  'gamma':['scale', 'auto'],
                  'kernel': ['rbf']}

    grid = GridSearchCV(svc, param_grid, scoring = 'accuracy')
    grid.fit(X_train, y_train)
    grid_predictions = grid.predict(X_test)
    print(grid.best_params_)

    #Predict values based on new parameters
    y_pred_acc = grid.predict(X_test)
    #add metrcis to df
    new_row = {'Model':"Sentiment + SVM",'Features':list(feature_df.iloc[:,feature_list].columns.values),'Accuracy Score':accuracy_score(y_test,y_pred_acc), 'Precision Score':precision_score(y_test,y_pred_acc), 'Recall Score':recall_score(y_test,y_pred_acc), 'F1 Score ':f1_score(y_test,y_pred_acc)}
    figure_3_df= figure_3_df.append(new_row, ignore_index=True)

    return figure_3_df
def normalize(df):
    scaler=StandardScaler()
    for column in df.columns[1:12]:
        scaled_data = scaler.fit_transform(df[[column]])
        df[column] = scaled_data
    return df
########Main##########
#prepare the dataset with technical features only. Using self-choosen features --> same day return instead of previou's days return
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Ticker_TSLA.csv')
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.dropna(axis=0, how="any")
Feature_set_Ticker_TSLA = add_alpha_boolean(Feature_set_Ticker_TSLA)
Feature_set_Ticker_TSLA =  Feature_set_Ticker_TSLA.drop(columns=['next day return', "previous day's return", 'open', 'close'])
Feature_set_Ticker_TSLA = normalize(Feature_set_Ticker_TSLA)


figure_3 = pd.DataFrame([], columns=['Model','Features','Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score '])
figure_3 = baseline([1,2,3,4], Feature_set_Ticker_TSLA, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility']
figure_3 = sentiment_logreg([1,2,3,4,5,6,11], Feature_set_Ticker_TSLA, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Stanford Sentiment']
figure_3 = sentiment_svm([1,2,3,4,5,6,11], Feature_set_Ticker_TSLA, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Stanford Sentiment']


#Same procedure for product set
#prepare the dataset with technical features only. Using self-choosen features --> same day return instead of previou's days return
Feature_set_Product_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Product_TSLA.csv')
Feature_set_Product_TSLA = Feature_set_Product_TSLA.dropna(axis=0, how="any")
Feature_set_Product_TSLA = add_alpha_boolean(Feature_set_Product_TSLA)
Feature_set_Product_TSLA =  Feature_set_Product_TSLA.drop(columns=['next day return', "previous day's return", 'open', 'close'])
Feature_set_Product_TSLA = normalize(Feature_set_Product_TSLA)

new_row = {'Features': "TESLA product set"}
figure_3= figure_3.append(new_row, ignore_index=True)
figure_3 = baseline([1,2,3,4], Feature_set_Product_TSLA, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility']
figure_3 = sentiment_logreg([1,2,3,4,5,6,11], Feature_set_Product_TSLA, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility']
figure_3 = sentiment_svm([1,2,3,4,5,6,11], Feature_set_Product_TSLA, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum']

#Same procedure for product set with Imputed features
#prepare the dataset with technical features only. Using self-choosen features --> same day return instead of previou's days return
Feature_set_Product_TSLA_imp = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Product_TSLA_imp.csv')
Feature_set_Product_TSLA_imp = Feature_set_Product_TSLA_imp.dropna(axis=0, how="any")
Feature_set_Product_TSLA_imp = add_alpha_boolean(Feature_set_Product_TSLA_imp)
Feature_set_Product_TSLA_imp =  Feature_set_Product_TSLA_imp.drop(columns=['next day return', "previous day's return", 'open', 'close'])
Feature_set_Product_TSLA_imp = normalize(Feature_set_Product_TSLA_imp)

new_row = {'Features': "TESLA product set with imputed features"}
figure_3= figure_3.append(new_row, ignore_index=True)
figure_3 = baseline([1,2,3,4], Feature_set_Product_TSLA_imp, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility']
figure_3 = sentiment_logreg([1,2,3,4,5,6,11], Feature_set_Product_TSLA_imp, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility']
figure_3 = sentiment_svm([1,2,3,4,5,6,11], Feature_set_Product_TSLA_imp, figure_3) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum']

pd.set_option('display.max_columns', None)
print(figure_3)
figure_3.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/1. Output Daily/alpha_predictions_figure_3.csv', index = False)
