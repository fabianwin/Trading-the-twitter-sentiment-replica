#########Notes############
"""
1. Predict binary outcome: will stock return be positive or negative. Use 2 different models from paper with paper choosen sentiment&technical features
    1.1 Support Vector Machine
    1.2 Logistic Regression
"""
########Libraries########
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


########Functions########
def add_alpha_boolean(df):
    for i, row in df.iterrows():
        if row['same_hour_return'] >= 0:
            sig = 1
        else:
            sig=0
        df.at[i, 'positive alpha'] = sig
    return df

#prepare the dataset
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/2. Data Hourly/Feature_set_Ticker_TSLA.csv')
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.dropna(axis=0, how="any")
Feature_set_Ticker_TSLA = add_alpha_boolean(Feature_set_Ticker_TSLA)
Feature_set_Ticker_TSLA =  Feature_set_Ticker_TSLA.drop(columns=['next_hour_return', 'previous_hour_return', 'open', 'close'])
X_train, X_test, y_train, y_test = train_test_split(Feature_set_Ticker_TSLA.iloc[:,[1,2,3,4,5,8,9]], Feature_set_Ticker_TSLA.iloc[:,10], test_size=0.3,random_state=42)

########SVM##########
#Create a svm Classifier with RBF Kernel
rbf_svc = svm.SVC(C=100, gamma='scale', kernel='rbf', verbose=True)
#Train the model using the training sets
rbf_svc.fit(X_train, y_train)
#Predict the response for test dataset
y_svc_pred = rbf_svc.predict(X_test)

predictions = pd.DataFrame(y_svc_pred, columns= ['y_svc_pred'])
predictions['y_test']=y_test.values
print(predictions)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_svc_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_svc_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_svc_pred))
print("---------------------------------------")


########Logistic regression##########
#define logistic regression parameters
logreg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)

#train model and make prediction
logreg.fit(X_train, y_train)
y_logreg_pred = logreg.predict_proba(X_test)

#build dataframe for data analysis
predictions_logreg = pd.DataFrame(y_logreg_pred, columns= ['y_logreg_pred_0', 'y_logreg_pred_1'])
predictions_logreg["y_predict"] = np.nan
for i, row in predictions_logreg.iterrows():
    if row['y_logreg_pred_0'] >= 0.5:
        row["y_predict"] = 0
    else:
        row["y_predict"] = 1
predictions_logreg['y_test']=y_test.values
print(predictions_logreg)

#get metrics score
y_logreg_pred = predictions_logreg['y_predict']
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_logreg_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_logreg_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_logreg_pred))
print("---------------------------------------")


########Logistic regression##########
