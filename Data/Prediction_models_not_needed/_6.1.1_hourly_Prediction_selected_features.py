#########Notes############
"""
Learn which sentiment features improve the accuracy.

1. create baseline model: logisitc linear regression with technical/financial features only
2. find which sentiment features increases the accuracy
"""
########Libraries########
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd


########Functions########
def add_alpha_boolean(df):
    for i, row in df.iterrows():
        if row['same_hour_return'] >= 0:
            sig = 1
        else:
            sig=0
        df.at[i, 'positive alpha'] = sig
    return df

########Main##########
#prepare the dataset
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/2. Data Hourly/Feature_set_Ticker_TSLA.csv')
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.dropna(axis=0, how="any")
Feature_set_Ticker_TSLA = add_alpha_boolean(Feature_set_Ticker_TSLA)
Feature_set_Ticker_TSLA =  Feature_set_Ticker_TSLA.drop(columns=['next_hour_return', 'previous_hour_return', 'open', 'close'])
print(Feature_set_Ticker_TSLA.dtypes)
print(Feature_set_Ticker_TSLA.iloc[:,[1,8,9]])
X_train, X_test, y_train, y_test = train_test_split(Feature_set_Ticker_TSLA.iloc[:,[1,2,3,4,5,8,9]], Feature_set_Ticker_TSLA.iloc[:,10], test_size=0.3,random_state=42)

"""
#Create a svm Classifier with RBF Kernel
rbf_svc = svm.SVC(C=100, gamma='scale', kernel='rbf', verbose=True)
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

#----------
Train_pred = rbf_svc.predict(X_train)
print(Train_pred)
"""
