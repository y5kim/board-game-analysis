import ast
import os
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Functions used in analysis
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

## Functions used in recommender
def get_reversed_encodings(encodings):
    """
    Returns dictionary with inverted key:value pairs
    
    encodings: dictionary
    """
    assert isinstance(encodings, dict)
    
    return {value:key for key, value in encodings.items()}

def get_encoded_vec(items, encodings):
    """
    Returns a multi-hot vector encoding corresponding to tokens in "items"

    items: list of tuple 
    encodings: dictionary
    """
    assert isinstance(items, (list, tuple))
    assert isinstance(encodings, dict)

    rev_encodings = get_reversed_encodings(encodings)
    encoded_vec = [0]*len(encodings.keys())
    for i in items:
        try:
            encoded_vec[rev_encodings[i]] = 1
        except:
            pass
        
    return encoded_vec

def add_encoded_column(df, col, threshold=0, filt_items=[]):
    """
    Adds a multi-hot encoded version of a multi-category column 'col' to 'df'
    and returns the corresponding encoding dict

    df: dataframe
    col: name of column to be encoded
    threshold: threshold on the number of occurrences 
    filt_items: name of items to be filtered out
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(col, str) and col in list(df)
    assert isinstance(threshold, int)
    assert isinstance(filt_items, list)

    # Count the number of occurrences for each item 
    cnt = Counter(itertools.chain.from_iterable(df[col]))
    item_set = {x[0] for x in cnt.most_common() if x[1] >= threshold and x[0] not in filt_items}

    item_encodings = dict(enumerate(sorted(item_set)))
    df[f'{col}_encoded'] = df[col].apply(get_encoded_vec, args=(item_encodings,))

    return item_encodings

