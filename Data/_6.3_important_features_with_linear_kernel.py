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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


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

#!!!!!!!
#The next row takes too many input variables for making it computational coefficient
X_train, X_test, y_train, y_test = train_test_split(Feature_set_Ticker_TSLA.iloc[:,[5,6,7,8,9,10,11,12]], Feature_set_Ticker_TSLA.iloc[:,16], test_size=0.3,random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(Feature_set_Ticker_TSLA.iloc[:,[7, 12]], Feature_set_Ticker_TSLA.iloc[:,16], test_size=0.3,random_state=42)


#find the suitable for multiple parameters
param_grid = {'C': [5**-3,10**-3,15**-3],
              'gamma':['scale', 'auto'],
              'kernel': ['linear','rbf']}
svc = svm.SVC()
grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3,n_jobs=-1) #verbose=3 for more info
# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)
# print classification report
grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))

#find the most relevant features
Feature_set_Ticker_TSLA.drop(["open","close","next day return","same day return","date","normalized average sentiment"], axis=1,inplace=True)
X = Feature_set_Ticker_TSLA.iloc[:,:10]
y = Feature_set_Ticker_TSLA.iloc[:,10]
svc = svm.SVC(C=0.008, kernel='linear', gamma='scale')
svc.fit(X, y)
importance = np.abs(svc.coef_)
importance = importance.reshape((10,))
feature_names = np.array(X.columns)

plt.bar(height=importance.astype(int), x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

#####findings
"""
Looking at the weights of the features we can see that 3 relevant feature exist:
1. number of tweets
2. price momentum
3. sentiment momentum

all the other features have weights of the size 10^-2 and hence are not playing a signifcant role
"""
