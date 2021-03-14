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
    assert isinstance(lb, int) and ub > 0
    assert isinstance(ub, int) and lb > 0
    assert isinstance(exponential_regression, bool)

    # Filter the dataframe on yearpublished lower bound and upper bound
    filtered_df = df.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub)]
    
    # Configure the pyplot setting
    fig = plt.figure(figsize=(15,10))
    sns.set(style="ticks")
    
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
    p = sns.histplot(filtered_df["yearpublished"], discrete=True, stat="count", color="orange", edgecolor="white")
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
    assert isinstance(lb, int) and lb > 0
    assert isinstance(ub, int) and ub > 0
    assert isinstance(min_colname, str) and min_colname in df.columns
    assert isinstance(max_colname, str) and max_colname in df.columns
    assert isinstance(ylabel_name, str)

    # Filter the dataframe on yearpublished lower bound and upper bound
    df = df.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub)]
    fig = plt.figure(figsize=(15,10))

    # Plot the line plot of average values with shades of 95% confidence interval
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
    assert isinstance(lb, int) and lb > 0
    assert isinstance(ub, int) and ub > 0
    assert isinstance(year_threshold, int) and 1900 <= year_threshold <= 2020

    # Filter the dataframe on yearpublished lower bound and upper bound
    filtered_df = df.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub)]
    attribute_list = ["numratings", "avgrating"]

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(18,13))
    # Plot the line plot of number of ratings
    p1 = sns.lineplot(data=filtered_df, x="yearpublished", y="numratings", ax=ax1, color="red", linewidth=2.5)
    p1.set_xlabel("Year", fontsize=25,weight="bold")
    p1.set_ylabel("Number of ratings", fontsize=25,weight="bold")
    p1.tick_params(labelsize=20) 
    # Plot the line plot of rating scores
    p2 = sns.lineplot(data=filtered_df, x="yearpublished", y="avgrating", ax=ax2, color="blue", linewidth=2.5)
    p2.set_xlabel("Year", fontsize=25, weight="bold")
    ax2.set_ylabel("Score", fontsize=25, weight="bold")
    p2.tick_params(labelsize=23)
    p2.set_xticks(p2.get_xticks()[1:-2])
    # Plot the vertical line on year_threshold
    if year_threshold is not None:
        ax2.axvline(year_threshold, linewidth=2, ls='--', color="black")
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.show()

def generate_word_cloud(ds, max_words=200, width=500, height=500, background_color='white', title=""):
    """
    Generate word clouds given a pandas series

    ds: data series
    max_words: maximum number of words to be displayed
    width: width of the plot
    height: height of the plot
    background_color: background color of the plot
    title: title of the plot
    """
    assert isinstance(ds, pd.Series)
    assert isinstance(max_words, int) and max_words > 0
    assert isinstance(width, int) and width > 0
    assert isinstance(height, int) and height > 0
    assert isinstance(background_color, str)
    assert isinstance(title, str)
        
    def get_stop_words_wordcloud():
        """
        Helper function to identify words to be exclude from the word cloud
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
    stop_words = get_stop_words_wordcloud()
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

def compare_top_ten_items(df, colname, color_font, stop_words, year_threshold=2007):

    """
    Plot 10 items that appear most frequently before and after year_threshold side by side
    Return split dataframes, item counter and cmmon items occurring before and after year_threshold
    
    df: dataframe
    colname: name of column to be compared
    color_font: flag of whether to highlight items not shown in common items before and after year_threshold
    stop_words: list of words to be ignored
    year_threshold: yearpublished threshold
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(colname, str) and colname in df.columns
    assert isinstance(color_font, bool)
    assert isinstance(stop_words, list) and all(isinstance(x, str) for x in stop_words)
    assert isinstance(year_threshold, int) and year_threshold > 0
    
    # Split up dataframes into before and after year_threshold
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
    val_lim = max(max(vals1), max(vals2))+5

    # Plot the barplots before & after yearthreshold side by side
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

def plot_changed_frequencies(common_items, item_cnts1, item_cnts2, population_size1, population_size2, year_threshold):
    """
    Plot the changed occurrences of common items between before and after a given year
    
    common_items: list of items to be compared
    item_cnts1: occurrences of items before year_threshold
    item_cnts2: occurrences of items after year_threshold
    population_size1: population size before year_threshold
    population_size2: population size after year_threshold
    year_threshold: publishedyear threshold
    """

    assert isinstance(common_items, list)
    assert isinstance(item_cnts1, Counter) and isinstance(item_cnts2, Counter)    
    assert all(isinstance(x, str) for x in common_items)
    assert all(x in item_cnts1 and x in item_cnts2 for x in common_items)  # Check that common_items exist in item_cnts
    assert isinstance(population_size1, int) and population_size1 > 0
    assert isinstance(population_size2, int) and population_size2 > 0
    assert isinstance(year_threshold, int) and 1900 <= year_threshold <= 2020
    
    # Get the normalized frequency of common items
    plt.figure(figsize=(10,3))
    d1 = [100*item_cnts1[k]/population_size1 for k in common_items]
    d2 = [100*item_cnts2[k]/population_size2 for k in common_items]
    
    # Plot scatter plots of normalized frequencies before and after year_threshold
    plt.scatter(d1, common_items, c="royalblue", s=110, label="Before {}".format(year_threshold))
    plt.scatter(d2, common_items, c="salmon", s=110, label="After {}".format(year_threshold))
    plt.xlabel("Share (%)", fontsize=15, weight="bold")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15, weight="bold")

    # Plot arrows starting from the value before year_threshold to the value after year_threshold
    for i in np.arange(len(common_items)):
        if abs(d2[i]-d1[i]) > 0.5:
            if d1[i] < d2[i]:
                plt.arrow(d1[i]+0.3, i, (d2[i]-d1[i])-0.8, 0, head_starts_at_zero=False, head_width=0.12,width=0.0015, color="black")
            else:
                plt.arrow(d1[i]-0.3, i, (d2[i]-d1[i])+0.8, 0, head_starts_at_zero=False, head_width=0.12,width=0.0015, color="black")
    plt.legend(loc="lower right", fontsize=15)
    plt.show()
