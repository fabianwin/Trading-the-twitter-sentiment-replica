import pandas as pd
import numpy as np
from datetime import datetime
from _4_Feature_functions import construct_sentiment_feature_set #, number_of_tweets, daily_average_sentiment, sentiment_volatility, sentiment_momentum
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ticker_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/ticker_set_TSLA.csv')
product_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/product_set_TSLA.csv')
ticker_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/ticker_set_GM.csv')
product_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/product_set_GM.csv')

ticker_set_TSLA = ticker_set_TSLA.dropna(axis=0, how="any")
product_set_TSLA = product_set_TSLA.dropna(axis=0, how="any")
ticker_set_GM = ticker_set_GM.dropna(axis=0, how="any")
product_set_GM = product_set_GM.dropna(axis=0, how="any")

#initiate the feature dataframes where we can input the different features
col =["open","close","next day return","same day return","previous day's return", "volume", "price momentum", "price volatility","sentiment volatility", "sentiment momentum"]
df = pd.DataFrame({'date': pd.date_range(start="2020-08-01",end="2021-08-29", freq='D')})
Feature_set = df.reindex(columns = df.columns.tolist() + col)

#construct the feature sets and save them
Feature_set_Ticker_TSLA = construct_sentiment_feature_set(ticker_set_TSLA, Feature_set, "TSLA")
Feature_set_Ticker_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Ticker_TSLA.csv', index = False)


Feature_set_Product_TSLA = construct_sentiment_feature_set(product_set_TSLA, Feature_set, "TSLA")
Feature_set_Product_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Product_TSLA.csv', index = False)

"""
Feature_set_Ticker_GM = construct_sentiment_feature_set(ticker_set_GM, Feature_set,"GM")
Feature_set_Ticker_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Ticker_GM.csv', index = False)


Feature_set_Product_GM = construct_sentiment_feature_set(product_set_GM, Feature_set, "GM")
Feature_set_Product_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily//Feature_set_Product_GM.csv', index = False)
"""
