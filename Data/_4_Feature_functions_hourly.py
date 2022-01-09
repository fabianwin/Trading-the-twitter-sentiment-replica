import numpy as np
import pandas as pd
from datetime import datetime, date, time, timezone
import pytz
from sklearn import preprocessing
#----------------------------
def construct_sentiment_feature_set(twitter_df, feature_df, ticker_str):

    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """

    if ticker_str == "TSLA":
        finance_short_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_short_TSLA.csv')
        finance_long_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_extended_TSLA.csv')

    if ticker_str ==  "GM":
        finance_short_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_short_GM.csv')
        finance_long_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_extended_GM.csv')


    #get number of tweets
    feature_df = number_of_tweets(twitter_df, feature_df)
    #get daily average score
    feature_df = hourly_average_sentiment(twitter_df, feature_df, "Stanford Sentiment")
    feature_df = hourly_average_sentiment(twitter_df, feature_df, "TextBlob Sentiment")
    feature_df = hourly_average_sentiment(twitter_df, feature_df, "Flair Sentiment")
    feature_df = normalized_average_sentiment(feature_df)
    #get sentiment volatility
    feature_df = sentiment_volatility(twitter_df, feature_df)
    #get sentiment momentum
    feature_df = sentiment_momentum(twitter_df, feature_df, 8)
    #add add_financials (open, close, volume)
    feature_df = add_financials(finance_long_df,feature_df)
    #get same day's return
    feature_df = same_hour_return(feature_df)
    #get same day's return
    feature_df = next_hour_return (feature_df)
    #get previous day's return
    feature_df = previous_hour_return(feature_df)
    #get price momentum
    feature_df = price_momentum(feature_df, 15)
    #get price volatility
    feature_df = price_volatility(feature_df)


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
    date_count= pd.DataFrame(data=twitter_df['date_medium'].value_counts())
    date_count.index = pd.to_datetime(date_count.index)
    date_count = date_count.rename(columns={'date_medium':'number_of_tweets'})
    date_count['date'] = date_count.index

    feature_df = pd.merge(feature_df, date_count, how='left', on='date')

    return feature_df
#----------------------------
def hourly_average_sentiment(twitter_df, feature_df, sentiment_str):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby('date_medium', as_index=False)[sentiment_str].mean()
    df.date_medium = pd.to_datetime(df.date_medium)
    feature_df = pd.merge(feature_df, df, how='left', left_on='date', right_on='date_medium')
    feature_df= feature_df.drop(['date_medium'], axis=1)

    return feature_df
#----------------------------
def normalized_average_sentiment(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = feature_df.iloc[:,[7,8,9]]
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
    unique_dates = twitter_df['date_medium'].unique()
    unique_dates = pd.DataFrame(data=unique_dates, columns=['date_medium'])
    for i, row in unique_dates.iterrows():
        std = twitter_df.loc[twitter_df['date_medium']==row['date_medium'],'Stanford Sentiment'].std()
        volatility = std**.5
        feature_df.loc[feature_df['date'] == row['date_medium'], ['sentiment volatility']] = volatility

    return feature_df
#----------------------------
def sentiment_momentum(twitter_df, feature_df, h):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: df_final, same shape as df but with the inputed features
    """
    for i, row in feature_df.iterrows():
        t_now = row['date']
        t_before = row['date'] - pd.Timedelta(hours=h)
        s_now = row['Stanford Sentiment']
        s_before = feature_df.loc[feature_df['date'] == t_before,'Stanford Sentiment'].max()
        if s_before == 0:
            p = float("NooooN")
        else:
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
    finance_df.time = pd.to_datetime(finance_df.time, utc=True)
    finance_df = finance_df.rename(columns={'time':'date'})
    finance_df = finance_df.drop(['high','low'], axis=1)
    feature_df = pd.merge(feature_df, finance_df, how='left', on='date')
    #date_count.index = pd.to_datetime(date_count.index)
    return feature_df
#----------------------------
def same_hour_return(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    feature_df.same_hour_return = feature_df.close/feature_df.open -1

    return feature_df
#----------------------------
def next_hour_return (feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    feature_df.next_hour_return = feature_df.same_hour_return.shift(periods=-1)

    return feature_df
#----------------------------
def previous_hour_return(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    feature_df.previous_hour_return = feature_df.same_hour_return.shift(periods=1)

    return feature_df
#----------------------------
def price_momentum(feature_df, h):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: feature_df, same shape as df but with the inputed features
    """

    df = pd.DataFrame([feature_df.date, feature_df.close]).transpose()
    df = df.dropna(axis=0, how='any')
    df['shifted_close'] = df.close.shift(periods=h)
    df['momentum'] = df.close - df.shifted_close
    df = df.drop(['close','shifted_close'], axis=1)
    feature_df = pd.merge(feature_df, df, how='left', on='date')

    return feature_df
#----------------------------
def price_volatility(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = feature_df
    #calculate log returns
    df['log returns'] = np.log(feature_df['close']/feature_df['close'].shift())
    df = df.groupby(pd.Grouper(key='date',freq='D')).std()
    #calcualte volatility based on 15 trading hours per day. Every day will have the same value, could be improved with minute data.
    df['volatility'] = df['log returns']*15**.5

    #merge the dataframes together and drop unnecessairy columns
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df = df.iloc[:,[13]]
    feature_df = pd.merge(feature_df, df, left_on=[feature_df['date'].dt.year, feature_df['date'].dt.month, feature_df['date'].dt.day], right_on=[df.index.year, df.index.month,df.index.day], how='left')
    feature_df = feature_df.drop(['key_0', 'key_1', 'key_2'], axis=1)

    return feature_df
