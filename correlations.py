import ast
import os
import itertools
from collections import Counter

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pre




def get_weight_corr(df, cnt):

    columns = ["averageweight","avgrating"]
    for col in cnt:
        columns.append(col[0])
    corr = df[columns].corr(method='spearman')
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True,fmt = ".1f")
    plt.title("Correlations between mechanics and weight")
    plt.show()

def get_average_weight_by_mechanic(df,cnt):
    weights = df["averageweight"]
    averageWeights = []
    columns = []
    for col in cnt:
        columns.append(col[0])
    for col in columns:
        avg = 0
        count = 0
        for row in range(len(col)):
            if(col[row]):
                avg+=weights[row]
                count+=1
        avg/=count
        averageWeights.append(avg)
    plt.scatter(averageWeights,columns)
    plt.xlabel("Average Weight")
    plt.title("Average Weight by Mechanic")
    plt.grid()
    plt.show()

def weightVSnumMechanics(df):
    avgweight_by_numMechanics = {}
    weights = df["averageweight"]
    mechanics = df["mechanic"]
    for i in range(len(weights)):
        try:
            (x,y) = avgweight_by_numMechanics[len(mechanics[i])]
            avgweight_by_numMechanics[len(mechanics[i])] = (x+weights[i],y+1)
        except:
            avgweight_by_numMechanics[len(mechanics[i])] = (weights[i],1)
    for key in avgweight_by_numMechanics.keys():
        (x,y) = avgweight_by_numMechanics[key]
        avgweight_by_numMechanics[key] = x/y
    plt.plot(list(avgweight_by_numMechanics.keys()),list(avgweight_by_numMechanics.values()),'ro')
    plt.xticks(range(20))
    plt.xlabel("Number of Mechanics in a Given Game")
    plt.ylabel("Average weight across all games with this many mechanics")
    plt.title("Average Weight by number of mechanics")
    plt.grid()
    plt.show()

def popularity_metric_correlations(df):
    sns.pairplot(df,y_vars = ["avgrating","bayesavgrating"],x_vars = ["numratings","numowned","numwants","numcomments","rank"])
    plt.show()

if __name__ == '__main__':
    #preprocessing stuff
    # Read data
    games = pd.read_csv("Data/games_detailed_info.csv", index_col=0) # review stats
    # 1. Remove columns with > 20% of NA values
    key_columns = pre.keep_columns_with_few_na(games)
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
    games = pre.parse_list_columns(games, list_colnames)

    # 5. Create new dataframes with binary columns of 20 popular items
    games_category, category_cnt = pre.create_df_with_binary_columns(games, "category", 20)
    games_mechanic, mechanic_cnt = pre.create_df_with_binary_columns(games, "mechanic", 20)
    games_designer, designer_cnt = pre.create_df_with_binary_columns(games, "designer", 20)
    games_publisher, publisher_cnt = pre.create_df_with_binary_columns(games, "publisher", 20)

    #correlations stuff
    get_weight_corr(games_mechanic,mechanic_cnt)
    get_average_weight_by_mechanic(games_mechanic,mechanic_cnt)
    weightVSnumMechanics(games)
    popularity_metric_correlations(games)
