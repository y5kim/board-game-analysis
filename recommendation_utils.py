import numpy as np
import pandas as pd
import itertools
from collections import Counter
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from word_cloud import get_stop_words

def get_reversed_encodings(encodings):
    '''returns dictionary with inverted key:value pairs'''
    assert isinstance(encodings, dict)
    
    return {value:key for key, value in encodings.items()}

def get_encoded_vec(items, encodings):
    '''returns a multi-hot vector encoding corresponding to tokens in "items"'''
    assert isinstance(items, list)
    assert isinstance(encodings, dict)

    rev_encodings = get_reversed_encodings(encodings)
    encoded_vec = [0]*len(encodings.keys())
    for i in items:
        try:
            encoded_vec[rev_encodings[i]] = 1
        except:
            pass
        
    return encoded_vec

def add_encoded_column(df, col, threshold=None, filt_items=None):
    '''
    Adds a multi-hot encoded version of a multi-category column 'col' to 'df'
    and returns the corresponding encoding dict
    '''
    assert isinstance(df, pd.DataFrame)
    assert col in list(df)

    filt_items = [] if filt_items is None else filt_items
    if threshold:
        cnt = Counter(itertools.chain.from_iterable(df[col]))
        item_set = {x[0] for x in cnt.most_common() if x[1] >= threshold and x[0] not in filt_items}
    else:
        item_set = {i for row in df[col] for i in row}
        
    item_encodings = dict(enumerate(sorted(item_set)))
    df[f'{col}_encoded'] = df[col].apply(get_encoded_vec, args=(item_encodings,))

    return item_encodings

def add_item_counts_column(df, col):
    '''
    Adds a counter column to dataframe 'df' for a given list column 'col'
    '''
    assert isinstance(df, pd.DataFrame)
    assert col in list(df)

    df[f'num_{col}'] = df[col].apply(lambda x: len(ast.literal_eval(x)) if not(pd.isna(x)) else 0)

def generate_similarity_matrix(df, col, col_type='one_hot'):
    '''
    Returns a matrix of similarity scores of size (n,n) where n is the number of rows in the df

    Args:
        df: pandas dataframe
        col: column name in df
        col_type: one of 'one_hot', 'scalar', 'text'
    '''
    assert isinstance(df, pd.DataFrame)
    assert col in list(df)
    assert col_type in ['one_hot', 'scalar', 'text']

    if col_type == 'text':
        # use term frequency–inverse document frequency to encode text
        # use cosine similarity metric to quantify similarity between text
        stop_words = get_stop_words()
        tfidf = TfidfVectorizer(stop_words=stop_words)
        df[col] = df[col].fillna('')
        tfidf_matrix = tfidf.fit_transform(df[col])
        similarity_matrix = cosine_similarity(tfidf_matrix)
    elif col_type == 'one_hot':
        # use cosine-similarity to quantify similarity between encoded vectors
        col_matrix = np.array(df[col].values.tolist())
        similarity_matrix = cosine_similarity(col_matrix)
    elif col_type == 'scalar':
        # use normalized euclidean distances to qualtify similarity between scalars
        col_matrix = np.expand_dims(np.array(df[col].values.tolist()), axis=1)
        distance_matrix = pairwise_distances(col_matrix, metric = 'euclidean')
        similarity_matrix = 1 - distance_matrix/np.amax(distance_matrix)
    
    return similarity_matrix

def get_row_idx(df, value, col='name'):
    '''
    Returns the row index where the value in 'col' column of dataframe 'df' is 'value'
    if the value exists; returns -1 otherwise
    '''
    assert isinstance(df, pd.DataFrame)
    assert col in list(df)

    try:
        idx = int(df.index[df[col] == value][0])
    except IndexError:
        idx = -1

    return idx

def get_idx_values(df, indices, col='name'):
    '''
    Returns a list of values at given 'indices' in the 'col' column of dataframe 'df'
    '''
    assert isinstance(df, pd.DataFrame)
    assert col in list(df)
    indices = [indices] if not isinstance(indices, list) else indices
    assert all(isinstance(i, int) for i in indices)

    return list(df.iloc[indices, df.columns.get_loc(col)])



def find_nearest_idx(idx, similarity_matrices, similarity_weights=None, num_games=5):
    '''
    Returns a list of indices corresponding to the 'num_games' nearest games compared to
    the game specified by 'idx'

    Args:
        idx: int representing the input game
        similarity_matrices: list of numpy arrays; each numpy array maps the similarity score between
                             all games with respect to a specific parameter
        similarity_weights: optional list of floats of the same size as 'similarity_matrices; 
                            each entry specifies the weightage given to the corresponding similarity matrix
        num_games: int that specifies the number of nearest neighbours to return
    '''
    assert isinstance(idx, int)
    assert idx >= 0
    assert isinstance(similarity_matrices, list)
    assert similarity_weights is None or len(similarity_matrices) == len(similarity_weights)
    assert isinstance(num_games, int)


    similarity_weights = [1/len(similarity_matrices) for _ in range(len(similarity_matrices))] if similarity_weights is None else similarity_weights
    similarity_weights = np.expand_dims(np.array(similarity_weights), axis=1)

    game_similarity_vectors = np.array([mat[idx,:] for mat in similarity_matrices])

    weighted_similarities = np.sum(similarity_weights*game_similarity_vectors, axis=0)

    scores = sorted(list(enumerate(weighted_similarities)), key = lambda i: i[1], reverse=True)

    #exclude the input game itself
    return [i[0] for i in scores[1:num_games+1]]


def get_similarity_matrices(df, similarity_cols, similarity_types):
    '''
    Returns a list of similarity matrices corresponding to the columns specified in 'similarity_cols'

    Args:
        df: pandas dataframe
        similarity_cols: list of column names in 'df' for which to generate similarity matrices
        similarity_types: list of column types (each entry is one of: 'one_hot', 'scalar', 'text');
                          same size as 'similarity_cols'
    '''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(similarity_cols, list)
    assert isinstance(similarity_types, list)
    assert len(similarity_cols) == len(similarity_types)

    return [generate_similarity_matrix(df, col, col_type) for col, col_type in zip(similarity_cols, similarity_types)]
    

def recommend_games(df, name, similarity_matrices, similarity_weights=None, num_games=5):
    '''
    Returns a list of 'num_games' game names that are the nearest neighbours to the game specified by 'name'
    '''
    assert isinstance(name, str)
    game_idx = get_row_idx(df, name, col='name')
    assert game_idx >= 0, "Game not found"

    nearest_games_idx = find_nearest_idx(game_idx, similarity_matrices, similarity_weights, num_games)

    return get_idx_values(df, nearest_games_idx, col='name')
    