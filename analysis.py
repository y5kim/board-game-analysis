import ast
import os
import itertools
from collections import Counter
import re
from string import punctuation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
import nltk
from wordcloud import WordCloud, STOPWORDS


def plot_published_games_over_years(df, lb, ub, exponential_regression=True):
    """
    Plot the overall #published games over years from lb to ub

    df: dataframe 
    lb: yearpublished lower bound
    ub: yearpublished upper bound
    exponential_regression: a flag whether to plot an exponential regression line
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(lb, int) >= 1900
    assert isinstance(ub, int) and ub <=2020
    assert isinstance(exponential_regression, bool)

    # Filter the dataframe on yearpublished lower bound and upper bound
    filtered_df = df.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub)]
    
    # Configure the pyplot setting
    fig = plt.figure(figsize=(15,10))
    
    # Draw a exponential regression line
    if exponential_regression:
        transformer = FunctionTransformer(np.log, validate=True)
        counts = filtered_df.groupby("yearpublished").count()["id"]
        x = np.arange(len(counts))[:,None]; y = counts[:,None]
        # Fit exponential model 
        y_trans = transformer.fit_transform(y) 
        regressor = LinearRegression()
        results = regressor.fit(x, y_trans)
        model = results.predict
        y_fit = model(x)
        plt.plot(x+lb, np.exp(y_fit), "k--", color="brown", linewidth=2) 

    # Plot the histogram of published games
    p = sns.histplot(filtered_df["yearpublished"], discrete=True, stat="count", color="orange")
    p.set_xlabel("Year", fontsize=25, weight="bold")
    p.set_ylabel("Number of games", fontsize=25, weight="bold")
    p.tick_params(labelsize=20)
    p.set_xticks(p.get_xticks()[1:-2])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.show()

def plot_min_max_attributes_over_years(df, lb, ub, min_colname, max_colname, ylabel_name):
    """
    Plot the average and 95% confidence interval of min & max attributes over years from lb to ub

    df: dataframe 
    lb: yearpublished lower bound
    ub: yearpublished upper bound
    min_colname: name associated with minimum values
    max_colname: name associated with maximum values
    ylabel_name: label on y-axis
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(lb, int) >= 1900
    assert isinstance(ub, int) and ub <=2020
    assert isinstance(min_colname, str) and min_colname in df.columns
    assert isinstance(max_colname, str) and max_colname in df.columns
    assert isinstance(ylabel_name, str)


    df = games.loc[(games["yearpublished"] >= lb) & (games["yearpublished"] <= ub)]
    fig = plt.figure(figsize=(15,10))

    sns.lineplot(data=df, x="yearpublished", y=max_colname, label="Max", color="red", linewidth=2.5)
    p = sns.lineplot(data=df, x="yearpublished", y=min_colname, label="Min", color="blue", linewidth=2.5)
    plt.xlabel("Year", fontsize=25, weight="bold")
    plt.ylabel(ylabel_name, fontsize=25, weight="bold")
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=23, loc="upper left")
    p.set_xticks(p.get_xticks()[1:-2])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()

def plot_ratings_over_years(df, lb, ub, year_threshold=None):
    """
    Plot the average and 95% confidence interval of number and score of ratings over years from lb to ub

    df: dataframe 
    lb: yearpublished lower bound
    ub: yearpublished upper bound
    year_threshold: yearpublished threshold to mark in the plot
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(lb, int) >= 1900
    assert isinstance(ub, int) and ub <=2020 
    assert isinstance(year_threshold, int) and 1900 <= year_threshold <= 2020

    lb = 1990; ub = 2019
    filtered_df = df.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub)]
    attribute_list = ["numratings", "avgrating"]

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(18,13))
    p1 = sns.lineplot(data=filtered_df, x="yearpublished", y="numratings", ax=ax1, color="red", linewidth=2.5)
    p1.set_xlabel("Year", fontsize=25,weight="bold")
    p1.set_ylabel("Number of ratings", fontsize=25,weight="bold")
    p1.tick_params(labelsize=20) 
    p2 = sns.lineplot(data=filtered_df, x="yearpublished", y="avgrating", ax=ax2, color="blue", linewidth=2.5)
    p2.set_xlabel("Year", fontsize=25, weight="bold")
    ax2.set_ylabel("Score", fontsize=25, weight="bold")
    p2.tick_params(labelsize=23)
    p2.set_xticks(p2.get_xticks()[1:-2])
    if year_threshold is not None:
        ax2.axvline(year_threshold, linewidth=2, ls='--', color="black")
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.show()

def compare_top_ten_items(df, colname, color_font=False, stop_words = ["Card Game"], year_threshold=2007):
    """
    TBD
    """
    
    df1 = df.loc[df["yearpublished"] < year_threshold]
    df2 = df.loc[df["yearpublished"] >= year_threshold]
    
    # Count the occurrences of items 
    item_cnts1 = Counter(itertools.chain.from_iterable([y.strip("'").strip('"') for y in items.split(', ') 
                                                if len(y)>0 and y not in stop_words] for items in df1[colname]))
    item_cnts2 = Counter(itertools.chain.from_iterable([y.strip("'").strip('"') for y in items.split(', ') 
                                                if len(y)>0 and y not in stop_words] for items in df2[colname]))
    # Identify top 10 items by occurrences
    commons1 = item_cnts1.most_common(10)
    keys1 = [x[0].replace(" / ", "/") for x in commons1][::-1]
    vals1 = [100*x[1]/df1.shape[0] for x in commons1][::-1]

    commons2 = item_cnts2.most_common(10)
    keys2 = [x[0].replace(" / ", "/") for x in commons2][::-1]
    vals2 = [100*x[1]/df2.shape[0] for x in commons2][::-1]
    inters = [x for x in keys1 if x in keys2]
    val_lim = max(max(vals1), max(vals2))+4

    # Plot the barplots
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4.5))

    ax1.barh(keys1, vals1, color="royalblue")
    ax1.set_xlim((0,val_lim))
    ax1.set_xlabel("Share (%)", fontsize=18, weight="bold")
    ax1.yaxis.set_tick_params(labelsize=18)
    ax1.set_yticklabels(keys1, weight="bold")
    ax1.set_title("Before {}".format(year_threshold), fontsize=25, weight="bold", pad=20)
    for i, v in enumerate(vals1):
        ax1.text(v + 0.5, i-0.1, str(round(v,1)), color='royalblue', fontweight='bold', fontsize=15)
    ax1.set_xticks([])

    ax2.barh(keys2, vals2, color="salmon")
    ax2.set_xlim((0,val_lim))
    ax2.set_xlabel("Share (%)", fontsize=18, weight="bold")
    ax2.yaxis.set_tick_params(labelsize=18)
    ax2.set_title("After {}".format(year_threshold), fontsize=25, weight="bold", pad=20)
    ax2.set_xticks([])
    ax2.set_yticklabels(keys2, weight="bold")
    ax2.yaxis.tick_right()
    ax2.invert_xaxis()
    for i, v in enumerate(vals2):
        ax2.text((v + 3), i-0.1, str(round(v,1)), color='salmon', fontweight='bold', fontsize=15)
    
    # If color_font, highlight the items not commonly shown between the two dataframes
    if color_font:
        colors1 = ["royalblue" if x.get_text() not in inters else "black" for x in ax1.get_yticklabels()]
        [t.set_color(i) for (i,t) in
         zip(colors1, 
             ax1.get_yticklabels())]

        colors2 = ["salmon" if x.get_text() not in inters else "black" for x in ax2.get_yticklabels()]
        [t.set_color(i) for (i,t) in
         zip(colors2, 
             ax2.get_yticklabels())]
    plt.tight_layout()
    return(df1, df2, item_cnts1, item_cnts2, inters)

def generate_word_cloud(ds, max_words=200, width=500, height=500, background_color='white', title=""):
    """
    Generate word clouds
    """
    assert isinstance(ds, pd.Series)
        
    def get_stop_words():
        """
        Identify words to exclude from the word cloud
        """
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.extend([c for c in punctuation])
        stop_words.extend(['quot', 'rsquo', 'mdash', 'ndash','s', 'game', "", 
                          "playing", "play", "player", "games", "rule", "players",
                          "rule","turn","card", "win", "cards", "boards", "board",
                          "first", "played", "must", "may", "rules"])
        stop_words.extend([str(x) for x in np.arange(100)] +
                          ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])
        return set(stop_words)
    
    # Clean up words by excluding stop words
    stop_words = get_stop_words()
    tokenized_ds = ds.dropna().apply(nltk.word_tokenize)
    tokenized_ds = [ (re.sub(r'[^\w\s]','',x)).lower() for ls in tokenized_ds for x in ls ]
    words = [word for word in tokenized_ds if word not in stop_words]
    
    # Generate word cloud
    wc = WordCloud(background_color=background_color, max_words=max_words, width=width, height=height)
    wc.generate(' '.join(words))
    
    # Plot the word cloud
    fig=plt.gcf()
    fig.set_size_inches(15,10)
    plt.imshow(wc)
    plt.axis('off')   
    plt.title(title, fontsize=25, fontweight="bold", pad=20)
    plt.show()