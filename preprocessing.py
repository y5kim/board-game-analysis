import ast
import os
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def keep_columns_with_few_na(df, na_ratio_threshold=0.2):
    """
    Return columns of dataframe whose ratio of NA values below a given threshold

    df: Dataframe of games
    na_ratio_threshold: threshold on the number of NA values
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(na_ratio_threshold, float) and 0 <= na_ratio_threshold <= 1

    n_rows, n_cols = df.shape
    # Identify columns with NA values less than the threshold
    key_columns = [colname for colname in df.columns if df[colname].isna().sum() <= na_ratio_threshold*n_rows]
    return(key_columns)

def parse_list_columns(df, colnames):
    """
    Convert columns whose values are of string of list format to lists

    df: Dataframe of games
    colnames: list of column names to be parsed
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(colnames, list) and all(isinstance(i, str) and i in df.columns for i in colnames)
    
    # Convert the columns of string to list columns of list
    for list_col in colnames:
        df[list_col] = df[list_col].apply(lambda x: ast.literal_eval(x) if not(pd.isna(x)) else [])
    return(df)

def create_df_with_binary_columns(df, colname, n_binary_cols):
    """
    Create a new datframe where x most frequent items in the list turn into binary columns

    df: games Dataframe
    colname: name of the column to be coded
    n_binary_cols: top n columns to be generated into binary columns 
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(colname, str) and colname in df.columns
    assert isinstance(n_binary_cols, int) and n_binary_cols >=1

    new_df = df.copy()
    # Count the number of occurrences of a give column values
    cnt = list(itertools.chain.from_iterable(df[colname]))
    cnt = Counter(cnt)
    # Identify the most frequent items
    common_items = [x[0] for x in cnt.most_common(n_binary_cols)]
    # Create binary columns corresponding to the identified frequent items
    for col in common_items:
        new_df[col] = new_df[colname].apply(lambda x: col in x)
    return(new_df, cnt.most_common(n_binary_cols))

def clean_string_format_columns(df, colnames):
    """
    Clean up string-formated columns by removing "

    df: games Dataframe
    colnames: list of columns

    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(colnames, list) and all(isinstance(i, str) and i in df.columns for i in colnames)

    for colname in colnames:
        df[colname] = df[colname].apply(lambda x: x[1:-1] if isinstance(x, str) else "")
    return(df)


if __name__ == '__main__':
    # Read data
    games = pd.read_csv("Data/games_detailed_info.csv", index_col=0) # review stats
    # 1. Remove columns with > 20% of NA values 
    key_columns = keep_columns_with_few_na(games)
    # 2. Remove redundant/unnecesary columns
    unnecessary_columns = ["type", "thumbnail", "image", "suggested_num_players", "suggested_playerage", 
                           "suggested_language_dependence"]
    key_columns = [x for x in key_columns if x not in unnecessary_columns]
    # 3. Rename confusing column names
    games = games.loc[:,key_columns]
    games.rename(columns={"primary": "name", "usersrated": "numratings", "average": "avgrating",
                          "boardgamecategory": "category", "boardgamemechanic": "mechanic", 
                          "boardgamedesigner": "designer", "boardgamepublisher": "publisher", 
                          "bayesaverage": "bayesavgrating", "Board Game Rank": "rank", 
                          "stddev": "stdrating", "median": "medianrating",
                          "owned": "numowned", "trading": "numtrades", "wanting":"numwants", 
                          "wishing": "numwishes"}, inplace=True)
    # 4. Parse columns with list values
    list_colnames = ["category", "mechanic", "designer", "publisher"]
    games = parse_list_columns(games, list_colnames)

    # 5. Create new dataframes with binary columns of 20 popular items
    games_category, category_cnt = create_df_with_binary_columns(games, "category", 20)
    games_mechanic, mechanic_cnt = create_df_with_binary_columns(games, "mechanic", 20)
    games_designer, designer_cnt = create_df_with_binary_columns(games, "designer", 20)
    games_publisher, publisher_cnt = create_df_with_binary_columns(games, "publisher", 20)
