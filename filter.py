from matplotlib import image
from matplotlib.pyplot import delaxes
import pandas as pd
from pandas.core.algorithms import isin
import preprocessing as pre

def rating_filter(df, colname, keywords, random = False):
    '''
    filtering by keyword, and sort new df by ratings

    :param df(pandas Dataframe): the pre-processed games dataframe
    :param colname(str): the columns to be filtered
    :param keywords(str): the keywords to be filtered, separating by comma space.

    :return type: dataframe containing sorted filtered items
    '''

    assert isinstance(df, pd.DataFrame)
    assert isinstance(colname, str)
    assert isinstance(keywords, str)
    assert colname in ['designer', 'category', 'mechanic', 'publisher']

    #split the input keywords into list
    keys = keywords.split(', ')

    #filtering out the lists in the column taht contains the keywords
    mask = df[colname].apply(lambda x: all(item in x for item in keys))       #make a mask to check if keywords contained in each row
    out_df = df[mask]

    #if random is set to true, shuffle the out_df and return
    if random == True:
        out_df=out_df.sample(frac=1).reset_index(drop=True)             
        return out_df
    else:
        return out_df.sort_values('avgrating', ascending = False).reset_index(drop=True)       #default return in descending order


def select_next_n(filtered_games, n = 5):
    '''
    Select next n games

    :param df(pandas Dataframe)
    :param n(int) >=1

    :generating dataframe from next n down the index
    '''
    assert isinstance(filtered_games, pd.DataFrame)
    assert isinstance(n, int)
    assert n>=1

    #yield next n from the dataframe if current index not exceeding rows, and stop the generator at last iteration.
    curr_index = 0

    while curr_index < filtered_games.shape[0]:

        # if curr_index + n > filtered_games.shape[0]:            #stop iteration when reaching the end
        #     return filtered_games.iloc[curr_index:].reset_index(drop = True)

        yield filtered_games.iloc[curr_index:curr_index+n].reset_index(drop = True)      #yield next n games
        curr_index += n         #increment current index by n


def available_choices(filtered_games, colname):
    '''
    This function checks what are avaliable choices in the given column
    '''

    import itertools as it
    
    x = list(it.chain.from_iterable(filtered_games[colname]))


    return set(x)



def simple_filter():
    a = ''
    while a != "quit":
        print('Select from: designer, mechanic, category, publisher')
        print("enter \"quit\" to exit")
        a = input("Select type from above: ")

    pass


def display_top_words(filtered_games, colname, n = 5):
    '''
    This function displays the current top n keywords in the given column
    '''
    assert isinstance(filtered_games, pd.DataFrame) and not(filtered_games.empty)
    assert isinstance(n, int) and n>=1


    import itertools as it
    from collections import Counter

    x = list(it.chain.from_iterable(filtered_games[colname]))
    c = Counter(x)

    return 'The top {} {} are: '.format(n, colname)  + ', '.join(list(c.keys())[:n])


def image_from_url(url):

    #read image from url
    from PIL import Image
    import requests

    im = Image.open(requests.get(url, stream=True).raw)

    return im



def display_info(filtered_games):
    '''
    display info about the current dataframe
    '''

    assert isinstance(filtered_games, pd.DataFrame)
    
    for i in range(filtered_games.shape[0]):

        print('Game name: {}\n\n'.format(filtered_games['name'][i]),
            'Description: {}\n\n'.format(filtered_games['description'][i].replace('&#10;', '').replace('&#9;', '')),
            'The average rating among {} votes is {:.2f} out of 10\n'.format(filtered_games['numratings'][i], filtered_games['avgrating'][i]),
            'Among {} people, the difficulty rating is {:.2f} out of 5\n'.format(filtered_games['numweights'][i], filtered_games['averageweight'][i])
        
        )






if __name__ == '__main__':
    #preprocessing stuff
    # Read data
    games = pd.read_csv("Data/games_detailed_info.csv", index_col=0) # review stats
    # 1. Remove columns with > 20% of NA values
    key_columns = pre.keep_columns_with_few_na(games)
    # 2. Remove redundant/unnecesary columns
    unnecessary_columns = ["type", "suggested_num_players", "suggested_playerage",
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
    games = pre.parse_list_columns(games, list_colnames)



