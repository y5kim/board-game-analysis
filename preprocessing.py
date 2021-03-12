import ast
import os
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TODO: change the function name to identify_sparse_columns
def keep_columns_with_few_na(df, na_ratio_threshold=0.2):
    """
    Return columns of dataframe whose ratio of NA values <= threshold
    """
    n_rows, n_cols = df.shape
    key_columns = [colname for colname in df.columns if df[colname].isna().sum() <= na_ratio_threshold*n_rows]
    return(key_columns)

# TODO: parse_columns_to_lists
def parse_list_columns(df, colnames):
    """
    Convert columns whose values are of string of list format to lists
    """
    for list_col in colnames:
        df[list_col] = df[list_col].apply(lambda x: ast.literal_eval(x) if not(pd.isna(x)) else [])
    return(df)

def create_df_with_binary_columns(df, colname, n_binary_cols):
    """
    Create a new datframe where x most frequent items in the list turn into binary columns
    """
    new_df = df.copy()
    cnt = list(itertools.chain.from_iterable(df[colname]))
    cnt = Counter(cnt)
    common_items = [x[0] for x in cnt.most_common(n_binary_cols)]
    for col in common_items:
        new_df[col] = new_df[colname].apply(lambda x: col in x)
    return(new_df, cnt.most_common(n_binary_cols))

def clean_string_format_columns(df, colnames):
    """
    Clean up string-formated columns by removing "
    """
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



