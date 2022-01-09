import seaborn as sns
import pandas as pd
import os


########Functions##########
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


#only features from paper are used
Feature_set_Ticker_TSLA.drop(["open","close","next day return","same day return","date"], axis=1,inplace=True)

y = Feature_set_Ticker_TSLA.iloc[:,9]
X = Feature_set_Ticker_TSLA.iloc[:,1:9]

plt = sns.pairplot(Feature_set_Ticker_TSLA, hue="positive alpha",palette="bright")
my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Graphs/Sentiment_Correlation') # Figures out the absolute path for you in case your working directory moves around.
my_file = 'pairplot.png'
plt.savefig(os.path.join(my_path, my_file))
