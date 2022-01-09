import numpy as np
import pandas as pd
from datetime import date
import datetime
from sklearn import preprocessing
#----------------------------

def construct_sentiment_feature_set(twitter_df, feature_df, ticker_str):

    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    #ETF (ticker: CARZ) data:
    ETF_short_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_short_CARZ.csv')
    ETF_long_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_extended_CARZ.csv')
    ETF_short_df['date'] = pd.to_datetime(ETF_short_df['date'])

    if ticker_str == "TSLA":
        finance_short_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_short_TSLA.csv')
        finance_long_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_extended_TSLA.csv')

    if ticker_str ==  "GM":
        finance_short_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_short_GM.csv')
        finance_long_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_extended_GM.csv')

    #get number of tweets
    feature_df = number_of_tweets(twitter_df, feature_df)
    #get daily average score
    feature_df = daily_average_sentiment(twitter_df, feature_df, "Stanford Sentiment")
    feature_df = daily_average_sentiment(twitter_df, feature_df, "TextBlob Sentiment")
    feature_df = daily_average_sentiment(twitter_df, feature_df, "Flair Sentiment")
    feature_df = normalized_average_sentiment(feature_df)
    #get sentiment volatility
    feature_df = sentiment_volatility(twitter_df, feature_df)
    #get sentiment momentum
    feature_df = sentiment_momentum(twitter_df, feature_df, 5)

    #add add_financials
    add_financials(finance_short_df,feature_df)
    #get same day's return
    same_day_return(finance_short_df, feature_df)
    #get same day's return
    next_day_return(finance_short_df, feature_df)
    #get previous day's return
    previous_day_return(finance_short_df, feature_df)
    #get daily volume
    volume(finance_short_df, feature_df)
    #get price momentum
    price_momentum(finance_short_df, feature_df, 5)
    #get price volatility
    price_volatility(finance_long_df, feature_df)
    #get alpha
    feature_df = alpha(feature_df, ETF_short_df)



    pd.set_option('display.max_columns', None)
    print(feature_df)
    pd.reset_option('display.max_rows')

    return feature_df

#Sentiment functions
#----------------------------
def number_of_tweets(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    date_count= pd.DataFrame(data=twitter_df['date_short'].value_counts())
    date_count.index = pd.to_datetime(date_count.index)
    date_count = date_count.rename(columns={'date_short':'number_of_tweets'})
    date_count['date'] = date_count.index
    feature_df = pd.merge(feature_df, date_count, how='left', on='date')

    return feature_df
#----------------------------
def daily_average_sentiment(twitter_df, feature_df, sentiment_str):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby('date_short', as_index=False)[sentiment_str].mean()
    df.date_short = pd.to_datetime(df.date_short)
    feature_df = pd.merge(feature_df, df, how='left', left_on='date', right_on='date_short')
    feature_df= feature_df.drop(['date_short'], axis=1)

    return feature_df
#----------------------------
def normalized_average_sentiment(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = feature_df.iloc[:,[10,11,12]]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    normalized_df = pd.DataFrame(x_scaled)
    normalized_df['normalized average sentiment'] = normalized_df.iloc[:, 0:2].mean(axis=1)
    normalized_df['date'] = feature_df['date']
    feature_df = pd.merge(feature_df, normalized_df[['date','normalized average sentiment']], how='left', on='date')

    return feature_df
#----------------------------
def sentiment_volatility(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    unique_dates = twitter_df['date_short'].unique()
    unique_dates = pd.DataFrame(data=unique_dates, columns=['date_short'])
    for i, row in unique_dates.iterrows():
        std = twitter_df.loc[twitter_df['date_short']==row['date_short'],'Stanford Sentiment'].std()
        volatility = std**.5
        feature_df.loc[feature_df['date'] == row['date_short'], ['sentiment volatility']] = volatility

    return feature_df
#----------------------------
def sentiment_momentum(twitter_df, feature_df, d):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: df_final, same shape as df but with the inputed features
    """
    for i, row in feature_df.iterrows():
        t_now = row['date']
        t_before = row['date'] - pd.Timedelta(days=d)
        s_now = row['Stanford Sentiment']
        #s_before = feature_df.loc[feature_df['date'] == t_before,'Stanford Sentiment'].max()
        x=0
        #iterate and make date intervall bigger until s_before is not zero and can be used in as denuminator in a division.
        while True:
            t_before_minus = t_before - pd.Timedelta(days=x)
            t_before_plus = t_before + pd.Timedelta(days=x)
            s_before = feature_df.loc[(feature_df['date'] >= t_before_minus) & (feature_df['date'] <= t_before_plus) ,'Stanford Sentiment'].mean()
            x +=1
            if s_before!=0:
                break
        p = (s_now / s_before)*100
        feature_df.loc[feature_df['date'] == row['date'], ['sentiment momentum']] = p

    return feature_df
#----------------------------
def sentiment_reversal(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    # TO DO

    return feature_df
#----------------------------
# Finance Functions
#---------------------
def add_financials(finance_df, feature_df):
    for i, row in finance_df.iterrows():
        feature_df.loc[feature_df['date'] == row['date'], ['open']] = row['1. open']
        feature_df.loc[feature_df['date'] == row['date'], ['close']] = row['4. close']

    return feature_df
#----------------------------
def same_day_return(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    #previous day return
    for i, row in finance_df.iterrows():
        rtn = row['4. close']/row['1. open']-1
        feature_df.loc[feature_df['date'] == row['date'], "same day return"]= rtn

    return feature_df
#----------------------------
def next_day_return(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    #previous day return
    finance_df['1. open'] = finance_df['1. open'].shift(periods=1)
    finance_df['4. close'] = finance_df['4. close'].shift(periods=1)
    for i, row in finance_df.iterrows():
        rtn = row['4. close']/row['1. open']-1
        feature_df.loc[feature_df['date'] == row['date'], "next day return"]= rtn

    return feature_df

#----------------------------
def previous_day_return(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    #previous day return
    finance_df['1. open'] = finance_df['1. open'].shift(periods=-2)
    finance_df['4. close'] = finance_df['4. close'].shift(periods=-2)
    for i, row in finance_df.iterrows():
        rtn = row['4. close']/row['1. open']-1
        feature_df.loc[feature_df['date'] == row['date'], "previous day's return"]= rtn

    return feature_df
#----------------------------
def volume(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed volume
    """
    #previous day return
    for i, row in finance_df.iterrows():
        feature_df.loc[feature_df['date'] == row['date'], "volume"]= row['6. volume']

    return feature_df
#----------------------------
def price_momentum(finance_df, feature_df, d):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: feature_df, same shape as df but with the inputed features
    """
    df_shifted = finance_df.shift(periods=d)
    df = (finance_df['4. close']-df_shifted['4. close']).to_frame()
    df['date'] = finance_df['date']
    for i,row in df.iterrows():
        feature_df.loc[feature_df['date'] == row['date'], ['price momentum']] = row['4. close']

    return df
#----------------------------
def price_volatility(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    finance_df['time'] = pd.to_datetime(finance_df['time'])
    finance_df = finance_df.groupby([finance_df['time'].dt.date]).std()
    finance_df['close'] = finance_df['close']**.5
    for i,row in finance_df.iterrows():
        feature_df.loc[feature_df['date'] == pd.Timestamp(i), ['price volatility']] = row['close']

    return feature_df
#----------------------------
def alpha(feature_df,ETF_short_df):
    for i, row in ETF_short_df.iterrows():
        rtn = row['4. close']/row['1. open']-1
        ETF_short_df.loc[ETF_short_df['date'] == row['date'], "ETF day return"]= rtn
    feature_df = pd.merge(feature_df, ETF_short_df.iloc[:,[8,9]], how='left', on='date')
    for i, row in feature_df.iterrows():
        alpha = row['same day return'] - row['ETF day return']
        feature_df.loc[feature_df['date'] == row['date'], 'Alpha'] = alpha
    return feature_df
