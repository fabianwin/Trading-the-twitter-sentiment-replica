import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import flair
from flair.data import Sentence
from textblob import TextBlob


def preprocess_tweet(tweet):
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#'
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove consequtive question marks
    tweet = re.sub('[?]+[?]', ' ', tweet)

    ## TODO: remove &amp - is HTML code for hyperlink

    return tweet

#---------------------
#Run this Cell to download the appropriate model and start it
import stanza
# Download an English model into the default directory
print("Downloading English model...")
stanza.download('en')

# Build an English pipeline, with all processors by default
print("Building an English pipeline...")
en_nlp = stanza.Pipeline('en')
#----------------------------------------------
def sentence_sentiment_df(doc):
    """
    - Parameters: doc (a Stanza Document object)
    - Returns: a mean sentiment score
    """
    sentiment_values = []
    for sentence in doc.sentences:
        sentiment_score = sentence.sentiment
        sentiment_values.append(sentiment_score)

    mean_sentiment = np.mean(sentiment_values)
    return mean_sentiment
#----------------------------------------------
def get_stanford_sentiment(text):
    """
    - Parameters: str
    - Returns: integer score
    """
    text = en_nlp(text)
    score = sentence_sentiment_df(text)

    return score
