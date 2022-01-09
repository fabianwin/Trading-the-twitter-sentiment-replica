import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
import os
import scipy.stats as stats
import pingouin as pg

#############Functions###################
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def normalize(df, col_name_str):
    data = df[col_name_str]
    data = (2*(data - np.min(data)) / (np.max(data) - np.min(data)))-1 #normalize between [-1,1]
    data = data*0.5 #normalize between [-0.5,0.5]
    new_col_name = col_name_str+" "+"normalized"
    df[new_col_name]=data
    return df

def get_coefficient_df(x_str,y_str, df):
    c = pg.corr(x=df[x_str],y=df[y_str])
    c['x'] = x_str
    c['y'] =y_str
    return c

def plot_sentiment_correlation(feature_df,ticker_str):
    df = normalize(feature_df, "Stanford Sentiment")
    df = normalize(feature_df, "TextBlob Sentiment")
    df = normalize(feature_df, "Flair Sentiment")

    g0 = sns.jointplot("Stanford Sentiment normalized", "next day return", data=df, kind='reg')
    c0 = get_coefficient_df("Stanford Sentiment normalized", "next day return", df)
    g1 = sns.jointplot("Stanford Sentiment normalized", "same day return", data=df, kind='reg')
    c1 = get_coefficient_df("Stanford Sentiment normalized", "same day return", df)
    g2 = sns.jointplot("Stanford Sentiment normalized", "previous day's return", data=df,kind='reg')
    c2 = get_coefficient_df("Stanford Sentiment normalized", "previous day's return", df)

    g3 = sns.jointplot("TextBlob Sentiment normalized", "next day return", data=df, kind='reg')
    c3 = get_coefficient_df("TextBlob Sentiment normalized", "next day return", df)
    g4 = sns.jointplot("TextBlob Sentiment normalized", "same day return", data=df, kind='reg')
    c4 = get_coefficient_df("TextBlob Sentiment normalized", "same day return", df)
    g5 = sns.jointplot("TextBlob Sentiment normalized", "previous day's return", data=df,kind='reg')
    c5 = get_coefficient_df("TextBlob Sentiment normalized", "previous day's return", df)

    g6 = sns.jointplot("Flair Sentiment normalized", "next day return", data=df, kind='reg')
    c6 = get_coefficient_df("Flair Sentiment normalized", "next day return", df)
    g7 = sns.jointplot("Flair Sentiment normalized", "same day return", data=df, kind='reg')
    c7 = get_coefficient_df("Flair Sentiment normalized", "same day return", df)
    g8 = sns.jointplot("Flair Sentiment normalized", "previous day's return", data=df,kind='reg')
    c8 = get_coefficient_df("Flair Sentiment normalized", "previous day's return", df)


    corr = pd.concat([c0, c1, c2, c3, c4, c5,c6,c7,c8], ignore_index=True)

    fig = plt.figure(figsize=(15,15))
    gs = gridspec.GridSpec(3, 3)
    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    mg2 = SeabornFig2Grid(g2, fig, gs[2])
    mg3 = SeabornFig2Grid(g3, fig, gs[3])
    mg4 = SeabornFig2Grid(g4, fig, gs[4])
    mg5 = SeabornFig2Grid(g5, fig, gs[5])
    mg6 = SeabornFig2Grid(g6, fig, gs[6])
    mg7 = SeabornFig2Grid(g7, fig, gs[7])
    mg8 = SeabornFig2Grid(g8, fig, gs[8])

    gs.tight_layout(fig)
    #gs.update(top=0.7)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Graphs/Sentiment_Correlation') # Figures out the absolute path for you in case your working directory moves around.
    my_file = ticker_str+""+"sentiment correlation"+'.png'
    plt.savefig(os.path.join(my_path, my_file))
    corr.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Graphs/Sentiment_Correlation/Pearson_scores.csv', index = False)
    # https://raphaelvallat.com/correlation.html to help analyse the data

    plt.show()

#############Main-Code###################
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Ticker_TSLA.csv')
plot_sentiment_correlation(Feature_set_Ticker_TSLA, "TSLA")
