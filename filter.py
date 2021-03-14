import itertools as it

import pandas as pd


def rating_filter(df, colname, keywords, random = False):
    """
    Filter a given dataframe by keyword, and sort new dataframe by ratings

    :param df: the pre-processed games dataframe
    :param colname: the columns to be filtered
    :param keywords: the keywords to be filtered, separating by comma space.

    :return type: dataframe containing sorted filtered items
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(colname, str)
    assert isinstance(keywords, str)
    assert colname in ['designer', 'category', 'mechanic', 'publisher']

    #split the input keywords into list
    keys = keywords.split(', ')

    #filtering out the lists in the column taht contains the keywords
    mask = df[colname].apply(lambda x: all(item in x for item in keys))  # make a mask to check if keywords contained in each row
    out_df = df[mask]

    #if random is set to true, shuffle the out_df and return
    if random == True:
        out_df=out_df.sample(frac=1).reset_index(drop=True)             
        return out_df
    else:
        return out_df.sort_values('avgrating', ascending = False).reset_index(drop=True)  # default return in descending order


def available_choices(filtered_games, colname):
    """
    Returns the available choices in the given column

    :param filtered_games: games dataframe
    :param colname: name of the column to be queried
    """
    assert isinstance(filtered_games, pd.DataFrame)
    assert isinstance(colname, str) and colname in filtered_games.columns

    return(set(it.chain.from_iterable(filtered_games[colname])))


def select_next_n(filtered_games, n=5):
    """
    Select next n games down the index from the given dataframe
    :param filtered_games: games dataframe
    :param n: number of games to be returned
    """
    assert isinstance(filtered_games, pd.DataFrame)
    assert isinstance(n, int)
    assert n >= 1

    # Yield next n from the dataframe if current index not exceeding rows, and stop the generator at last iteration.
    curr_index = 0
    while curr_index < filtered_games.shape[0]:
        yield filtered_games.iloc[curr_index:curr_index+n].reset_index(drop = True)  # Yield next n games
        curr_index += n  # Increment current index by n
