#########Notes############
"""
1. Predict binary outcome: will stock return be positive or negative
    1.1 L1 regularization
    1.2 RBF Kernel
    1.3 SVM
    1.4 use all technical and sentiment features from paper

2. try to split up the feature set to see if X-hours-windows can be predicted
"""
########Libraries########
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

########Functions########
def add_alpha_boolean(df):
    for i, row in df.iterrows():
        if row['same day return'] >= 0:
            sig = 1
        else:
            sig=0
        df.at[i, 'positive alpha'] = sig
    return df

########Main##########
#prepare the dataset
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Ticker_TSLA.csv')
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.dropna(axis=0, how="any")
Feature_set_Ticker_TSLA = add_alpha_boolean(Feature_set_Ticker_TSLA)
X_train, X_test, y_train, y_test = train_test_split(Feature_set_Ticker_TSLA.iloc[:,[5,6,7,8,9,10,11,12]], Feature_set_Ticker_TSLA.iloc[:,16], test_size=0.3,random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(Feature_set_Ticker_TSLA.iloc[:,[7, 12]], Feature_set_Ticker_TSLA.iloc[:,16], test_size=0.3,random_state=42)


print("find suitable parameters for a rbf kernel")
#find the suitable parameters  for a rbf kernel
param_grid = {'C': [0.35,0.4,0.45],
              'gamma':['scale', 'auto'],
              'kernel': ['rbf']}
svc = svm.SVC()
grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3,n_jobs=-1)
# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)

svc.fit(X_train, y_train)
perm_importance = permutation_importance(svc, X_train, y_train)


feature_names = ["previous day's return", 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets', 'Stanford Sentiment']
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()
