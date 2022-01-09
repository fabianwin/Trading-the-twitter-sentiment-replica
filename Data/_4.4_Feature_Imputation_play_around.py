"""
The entire code is copied and applied to the Ticker Dataset

Source: https://www.kaggle.com/datasets
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

#functions
def replaceWithNan(df, noOfCols, percOfNulls):
    """
    Description
    -----------
    This will randomly choose `noOfCols` columns and will randomly
    choose `percOfNulls` percent of the rows in each chosen column
    to replace the value of with NaN for a given dataframe `df`.

    Parameters
    ----------
    df (dataframe):
        The dataframe after converting the dictionary retrieved
        after calling function `appendValues()`.

    noOfCols (int):
        The number of columns you wish to randomly choose for
        NaN replacement.

    percOfNulls (int or float):
        The percentage of values in each chosen column that you
        want to replace with NaN.

    Returns
    -------
    ndf (dataframe):
        The dataframe with NaN replacements.

    Examples
    --------
    `Feature_set_Ticker_TSLA = replaceWithNan(Feature_set_Ticker_TSLA_original, 4, 25)` choose 4 columns at random
        and will choose 25% of the values to replace with NaN.
    `Feature_set_Ticker_TSLA = replaceWithNan(Feature_set_Ticker_TSLA_original, 7, 101)` will cause the machine to
        be confused af, and `Feature_set_Ticker_TSLA` will be `NoneType`.
    `Feature_set_Ticker_TSLA = replaceWithNan(Feature_set_Ticker_TSLA_original, 10000, .15)` will display a message
        saying that there aren't enough columns.
    """
    # If percentage is greater than 100%, then this won't work
    if percOfNulls > 100:
        print("You're probably asking to make more than the whole column null.")
        print("Why so confuse me? I am only machine! :(")
    # If the number of columns specified is more than the number of columns available,
    # then this print statement will be displayed
    elif noOfCols > len(df.columns):
        print(f"There are only {len(df.columns)} columns in your dataframe, but you're telling me to choose {noOfCols} columns.")
        print(f"If you want me to choose ALL columns, please set 'noOfCols' equal to {len(df.columns)}.")
    else:
        # If `percOfNulls` >=1, it will be assumed that `percOfNulls` is out of 100
        if percOfNulls >= 1:
            percOfNulls = percOfNulls/100
        # Make deep copy of original dataframe
        ndf = df.copy(deep = True)
        # For every randomly chosen column, replace `percOfNulls` percent of the values with NaN
        for column in ndf.sample(noOfCols, axis = 1):
            ndf[column] = ndf[column].sample(frac = 1 - percOfNulls)
        return ndf

def imputeMissingVals(ndf, imputer = IterativeImputer()):
    """
    Description
    -----------
    This will impute the NaNs in dataframe `ndf` with
    `IterativeImputer()`. If `imputer` is invalid, an error will
    be displayed along with the error description. If no imputer
    is specified in the function, the function will use the
    default imputer, `IterativeImputer()` with no specific
    parameters. The default estimator used for the imputer is
    `BayesianRidge()`.

    Parameters
    ----------
    ndf (dataframe):
        The dataframe containing NaNs, which was retrieved from
        function `replaceWithNan()`.

    imputer (IterativeImputer()):
        `IterativeImputer()` with any parameters of your choice.

    Returns
    -------
    ndft (dataframe):
        A dataframe with the imputed values.

    """
    # If `imputer` is invalid, an error will be thrown
    try:
        imputer.fit(ndf)
    except Exception as e:
        print("============================================================================")
        print(f"I wasn't able to iteratively impute with the given imputer:\n{str(imputer)}.")
        print(f"This was the error I've received from my master:\n\n{e}.")
        print("\nNOTE:\n\tAn imputer is already given by default for this function.")
        print("\tIf you don't set `imputer` to anything, then the default")
        print("\t`IterativeImputer()` with no other parameters will be used.")
        print("\tThe default estimator for `IterativeImputer()` is `BayesianRidge()`.")
        print("============================================================================")
    # If `imputer` is valid, then the imputer will fit to the dataframe containing NaNs and transform
    # A dataframe `ndft` will then be returned which contains the imputed values
    else:
        imputer.fit(ndf)
        ndft = pd.DataFrame(imputer.transform(ndf), columns = ndf.columns, index = ndf.index)

        return ndft

def displayResults(original_df, null_df, imputed_df, showDF = True, showGraph = True):
    """
    Description
    -----------
    This will display how far/close the imputed values were to
    the original values. Each column that was randomly chosen
    after running the function `replaceWithNan()` was a
    certain "time". For however many columns chosen, there will
    be that many dataframes created for comparison purposes.
    So basically, every column ("time") that has a NaN value in
    `null_df` will have it's own set of results which will be
    separated by a border: "===========".

    In each separation, there will be a dataframe of 4 rows.
    Each separation is it's own "time", and each separation
    will have different dates that were replaced by NaNs for
    the specific "time".
    So this means, a specific ("time", "date") pair was NaN.
    The first row is a row containing only NaNs from the
    dataframe `null_df`.
    The second row contains the original value for that given
    ("time", "date") pair.
    The third row contains the imputed value for that given
    pair.
    The fourth row contains the difference between the imputed
    and the original value.
    If the difference is positive, then it was an overestimation
    by the amount.
    If the difference is negative, the imputer underestimated
    the original value by the amount.The absolute value of the
    "differences" will be added up for each separation and will
    be displayed in each separation. The average difference
    will also be displayed in each separation. The average
    difference basically states that "for this current time, on
    an average, this was how far off the imputed value were from the
    original values." The overall average in difference will also be
    tracked to be displayed in the end along with the overall difference.
    A graph will also be displayed showing the difference between
    the original value and the imputed value for only the dates
    which were NaN for the current time in `null_df`.

    Parameters
    ----------
    original_df (dataframe):
        The original dataframe without any NaN replacements.
        This is the dataframe after converting the dictionary
        retrieved after calling the function `appendValues()`.

    null_df (dataframe):
        The dataframe after NaN replacements.
        This is the dataframe retrieved after calling the
        function `replaceWithNan()`.

    imputed_df (dataframe):
        The dataframe after imputations.
        This is the dataframe retrieved after calling the
        function `imputeMissingVals()`.

    showDF (bool):
        If True, then the dataframe containing the 4 rows
        will be displayed along with the other results.
        This is set to True by default.

    showGraph (bool):
        If True, then a graph will be displayed showing the
        original vs. imputed value.
        This is set to True by default.

    Returns
    -------
    Nothing.

    """
    # Keep track of the overall difference
    overallDifference = 0
    # Keep track of the overall average in difference
    overallAvg = 0

    cols = null_df.columns[null_df.isna().any()].tolist()
    for time in cols:

        # `currentDifference` will be used to display the overall difference for the current dataframe ("time")
        currentDifference = 0
        # `currentAvg` will be used to display the overall average in difference for the current dataframe ("time")
        currentAvg = 0

        # Print which "time" column we're dealing with
        print(f'Time:   {time}')

        # Create dataframe `somedf`, which contains 4 rows:
           # cdf[time]                         --> first row will have all values in our dataframe replaced with NaNs for the current "time" (index = 'the_nans')
           # adf[time]                         --> second row will have all values in our original dataframe for the current "time" (index = 'original_value')
           # round(cdft[time], 2)              --> third row will have all values (rounded) in our imputed dataframe for the current "time" (index = 'imputed_value')
           # round(cdft[time], 2) - adf[time]] --> fourth row will display the difference in imputed value to original value for the current "time" (index = 'difference')
             # Note: If difference is negative, that means the imputer underestimated the original value, if difference is positive, the imputer overestimated the original
        somedf = pd.DataFrame([null_df[time], original_df[time], round(imputed_df[time], 2), round(imputed_df[time], 2) - original_df[time]], index = ['the_nans', 'original_value', 'imputed_value', 'difference'])


        # Make the differences absolute value, and sum up the differences to get the total difference for the current "time"
        currentDifference = round(somedf.loc['difference'].abs().sum(), 2)
        print(f'Total difference for current time: {currentDifference}')

        # It's like an OnlyFans but for `onlyNans`
        onlyNans = somedf.loc[:, somedf.isna().any()]
        # Set `currentAvg` to the average of the absolute values of the differences
        currentAvg = round(onlyNans.loc['difference'].abs().mean(), 2)
        # Update `overallAvg`
        overallAvg += currentAvg
        print(f"Avg. difference for current time:  {currentAvg}")

        if (showDF):
            # Display dataframe of ONLY the dates containing NaNs for easier comparison
            display(onlyNans)

        if (showGraph):
            # Display a chart showing the difference between the original values
            # and the imputed values of ONLY the values which were NaNs
            plt.figure(figsize=(15.5, 4), dpi=80)
            plt.plot(onlyNans.loc['original_value'], color = 'green', marker = 'o', label="Original Values")
            plt.plot(onlyNans.loc['imputed_value'], color = 'red', marker = 'x', label="Imputed Values")
            plt.xticks(onlyNans.columns,rotation = 'vertical')
            plt.xlabel('NaN Dates', fontsize=12)
            plt.ylabel(time, fontsize=12)
            plt.title('Original vs Imputed', fontsize=14)
            plt.legend(loc="upper left")
            plt.show()

        # Update the overall difference
        overallDifference += currentDifference
        print('==============================================================================================================================================')

    # Print the overall difference, which is a measurement to determine how "off" the imputer was
    print(f'The overall difference is: ${round(overallDifference, 2)}')
    print(f'On an average, the imputer was ${round(overallAvg/len(cols), 2)} away from the actual value.')

#main
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Ticker_TSLA.csv')
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.dropna(axis=0, how="any")
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.set_index('date')
print(Feature_set_Ticker_TSLA.head())
Feature_set_Ticker_TSLA_original = Feature_set_Ticker_TSLA
Feature_set_Ticker_TSLA = replaceWithNan(Feature_set_Ticker_TSLA, 8, 0.15)
print(Feature_set_Ticker_TSLA.info())
print("")
print(f"Our dataframe containing null values has {Feature_set_Ticker_TSLA.count().sum()} values (not considering NaNs).")
print(f"Our original dataframe without null values has {Feature_set_Ticker_TSLA_original.count().sum()} values.")
print(f"This means that {Feature_set_Ticker_TSLA_original.count().sum() - Feature_set_Ticker_TSLA.count().sum()} values are NaN, which equates to approximately {((Feature_set_Ticker_TSLA_original.count().sum() - Feature_set_Ticker_TSLA.count().sum()) / (Feature_set_Ticker_TSLA_original.count().sum())*100):.04f}% of the dataset.")
print("")
print("the following columns have NaN values")
print(Feature_set_Ticker_TSLA.columns[Feature_set_Ticker_TSLA.isna().any()])

#imputing missing values with iterative imputer
Feature_set_Ticker_TSLA_imp = imputeMissingVals(Feature_set_Ticker_TSLA)

#display display
displayResults(Feature_set_Ticker_TSLA_original, Feature_set_Ticker_TSLA, Feature_set_Ticker_TSLA_imp)
