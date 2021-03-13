import numpy as np
import pandas as pd
import itertools
from collections import Counter
import ast
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from string import punctuation


def get_reversed_encodings(encodings):
    '''returns dictionary with inverted key:value pairs'''
    assert isinstance(encodings, dict)
    
    return {value:key for key, value in encodings.items()}

def get_encoded_vec(items, encodings):
    '''returns a multi-hot vector encoding corresponding to tokens in "items"'''
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

def get_stop_words():
    '''Returns a list of words to be ignored when searching for keywords'''
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend([c for c in punctuation] + ['quot', 'rsquo', 'mdash', 'ndash','s'])
    return set(stop_words)

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
    assert isinstance(similarity_cols, (list, tuple))
    assert isinstance(similarity_types, (list, tuple))
    assert len(similarity_cols) == len(similarity_types)

    return [generate_similarity_matrix(df, col, col_type) for col, col_type in zip(similarity_cols, similarity_types)]



def calculate_scores(idx, similarity_matrices, idx_weights=None, similarity_weights=None):
    '''
    Returns a tuple of (idx, score) corresponding to each entry in 'similarity_matrices'

    Args:
        idx: list of indices representing the input games
        idx_weights: weights determining the importance of each game
        similarity_matrices: list of numpy arrays; each numpy array maps the similarity score between
                             all games with respect to a specific parameter
        similarity_weights: optional list of floats of the same size as 'similarity_matrices; 
                            each entry specifies the weightage given to the corresponding similarity matrix
    '''
    assert isinstance(idx, (list, tuple))
    assert all(i >= 0 for i in idx)
    assert isinstance(similarity_matrices, (list, tuple))
    assert similarity_weights is None or len(similarity_matrices) == len(similarity_weights)
    assert idx_weights is None or len(idx_weights) == len(idx)

    similarity_weights = [1/len(similarity_matrices) for _ in range(len(similarity_matrices))] if similarity_weights is None else similarity_weights
    similarity_weights = np.expand_dims(np.array(similarity_weights), axis=1)

    idx_weights = [1/len(idx) for _ in idx] if idx_weights is None else idx_weights
    idx_weights = np.expand_dims(np.array(idx_weights), axis=1)

    game_similarity_vectors = np.array([np.sum(mat[idx,:] * idx_weights, axis=0) for mat in similarity_matrices])

    weighted_similarities = np.sum(similarity_weights*game_similarity_vectors, axis=0)

    scores = sorted(list(enumerate(weighted_similarities)), key = lambda i: i[1], reverse=True)

    return scores

def recommend_games(df, games, similarity_matrices, game_weights=None, similarity_weights=None, num_games=5, exclude=None):
    '''
    Returns a list of 'num_games' game names that are the nearest neighbours to the list of 'games' 
    excluding any items in 'exclude'

    Args:
        df: pandas dataframe containing board game data
        games: list of game names 
        game_weights: weights specifying preference of corresponding game; can be negative values
        similarity_matrices: list of numpy arrays; each numpy array maps the similarity score between
                             all games with respect to a specific parameter
        similarity_weights: optional list of floats of the same size as 'similarity_matrices; 
                            each entry specifies the weightage given to the corresponding similarity matrix
        num_games: number of recommended games to return
        exclude: list of items to exclude from recommendations
    '''
    assert isinstance(games, (list, tuple))
    assert isinstance(exclude, (list,tuple)) or exclude is None
    game_idx = [get_row_idx(df, game, col='name') for game in games]

    if game_weights is not None:
        try:
            #remove any game and associated weight if game was not found
            game_idx, game_weights = zip(*[(idx, weight) for idx, weight in zip(game_idx, game_weights) if idx>=0])
        except ValueError: #thrown if none of the games are found
            game_idx, game_weights = [],[]
    else:
        game_idx = [idx for idx in game_idx if idx >=0]

    assert len(game_idx) > 0, "Games not found"
    exclude_idx = [get_row_idx(df, game, col='name') for game in exclude] if exclude is not None else []

    game_scores = calculate_scores(game_idx, similarity_matrices, game_weights, similarity_weights)

    nearest_games = [score[0] for score in game_scores if score[0] not in list(game_idx)+list(exclude_idx)]

    return get_idx_values(df, nearest_games[:num_games], col='name')


def user_game_ratings(df, user):
    '''
    Returns a list of all game ratings for a given 'user'; returns empty list if user is not found
    '''
    assert isinstance(df, pd.DataFrame)

    return df[df['user'] == user][['name', 'rating']].values.tolist()

def rating_to_weights(ratings, mean=7.0):
    '''
    Converts rating values to a game-importance weight
    '''
    assert isinstance(ratings, (list, tuple))

    return [rating - mean for rating in ratings]