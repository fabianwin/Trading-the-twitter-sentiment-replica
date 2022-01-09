#Construct the Ticker Set for TSLA
import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from datetime import datetime
from _3_Sentiment_functions import preprocess_tweet, get_stanford_sentiment
from textblob import TextBlob
import flair
from flair.data import Sentence

#############Functions###################
def scrape_tweets_and_sentiments(max_tweets_int, keyword_str,twitter_df):
    """
    - Parameters: doc (a Stanza Document object)
    - Returns: a mean sentiment score
    """
    maxTweets = max_tweets_int
    keyword = keyword_str
    date = 'since:2020-08-01 until:2021-08-30'
    restrictions='min_faves:1 exclude:retweets lang:"en"' #min. 100 likes ,no retweets, in english

    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword +' '+ date+' '+restrictions).get_items()) :
            if i == maxTweets :
                break
            tmp = pd.Series([tweet.id,tweet.date.date(),tweet.date.replace(minute=0, second=0),tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount], index=col)
            tmp[5] = preprocess_tweet(tmp[5])
            if isinstance(tmp[5], str) == True:
                twitter_df = twitter_df.append( tmp, ignore_index=True)

    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    for i, row in twitter_df.iterrows():
        if isinstance(row['content'], str) == True:
                Stanford_sentiment = get_stanford_sentiment(row['content'])
                TextBlob_sentiment = TextBlob(row['content']).sentiment.polarity
                sentence = flair.data.Sentence(row['content'])
                sentiment_model.predict(sentence)
                if sentence.labels[0].value=="POSITIVE":
                    Flair_sentiment = sentence.labels[0].score
                else :
                    Flair_sentiment =1-sentence.labels[0].score
                twitter_df.at[i, 'Stanford Sentiment'] = Stanford_sentiment
                twitter_df.at[i, 'TextBlob Sentiment'] = TextBlob_sentiment
                twitter_df.at[i, 'Flair Sentiment'] = Flair_sentiment


    pd.set_option('display.max_columns', None)
    print(twitter_df)
    pd.reset_option('display.max_rows')

    return twitter_df
#########################################

col =["tweet_id","date_short","date_medium","date_long","username","content","likes","retweets","followers Count"]
colum = ["tweet_id","date_short","date_medium","date_long","username","content","likes","retweets","followers Count","Stanford Sentiment", "TextBlob Sentiment","Flair Sentiment"]
ticker_tweets = pd.DataFrame(columns=colum)


#Get Tesla tweets
ticker_tweets_TSLA = scrape_tweets_and_sentiments(1000000, "TSLA", ticker_tweets)
ticker_tweets_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/ticker_set_TSLA.csv', index = False)
ticker_tweets_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/2. Data Hourly/ticker_set_TSLA.csv', index = False)


#Get GM tweets
ticker_tweets_GM = scrape_tweets_and_sentiments(100, "GM", ticker_tweets)
ticker_tweets_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/ticker_set_GM.csv', index = False)
ticker_tweets_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/2. Data Hourly/ticker_set_GM.csv', index = False)

print("Ticker Set completely scraped")
