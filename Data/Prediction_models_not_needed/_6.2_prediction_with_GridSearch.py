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

X_train, X_test, y_train, y_test = train_test_split(Feature_set_Ticker_TSLA.iloc[:,[7, 12]], Feature_set_Ticker_TSLA.iloc[:,16], test_size=0.3,random_state=42)

param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'gamma':['scale', 'auto'],
              'kernel': ['linear','rbf']}
svc = svm.SVC()

grid = GridSearchCV(estimator=svc, param_grid=param_grid, refit = True, verbose = 3,n_jobs=-1)
# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)
grid_predictions = grid.predict(X_test)

# print classification report
print(classification_report(y_test, grid_predictions))



"""
#Train the model using the training sets
rbf_svc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = rbf_svc.predict(X_test)

predictions = pd.DataFrame(y_pred, columns= ['y_pred'])
predictions['y_test']=y_test.values
print(predictions)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
"""
