import ast
import os
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## Trend analysis
def count_plot_over_year(df, lb, ub):
    """
    plot the overall # published games over years from lb to ub

    df: dataframe 
    lb: year lower bound
    ub: year upper bound
    """
    plt.rcParams["axes.labelsize"] = 18
    sns.set(style="ticks")
    #f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},
    #                                   figsize=(10,8))
    x = df.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub), "yearpublished"]
    #sns.boxplot(x, ax=ax_box)
    #p = sns.histplot(x, ax=ax_hist, discrete=True, stat="count")
    p = sns.histplot(x, discrete=True, stat="count")
    
    #ax_box.set(yticks=[], xlabel="")
    #ax_hist.set(xlabel='Published year', ylabel='Count')
    p.set_xlabel("Publisehd year", fontsize=15)
    p.set_ylabel("Count", fontsize=15)
    p.tick_params(labelsize=15)
#    ax_hist.set_xlabel(xlabel="Published",fontsize=14)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    p.set_title("The number of games published from {} to {}".format(lb, ub), fontsize=18)
    plt.show()


def grouped_count_plot_over_year(df, lb, ub, items, group_name):
    """
    plot the overall # published games by group over years from lb to ub

    df: dataframe 
    lb: year lower bound
    ub: year upper bound
    items: group items to be considered
    group_name: name of the group
    """
    filtered = df.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub)]
    df2 = filtered[["yearpublished"] + items].groupby("yearpublished").sum().reset_index()
    cnt = filtered[["yearpublished", "id"]].groupby("yearpublished").count().reset_index()
    cnt.rename(columns={"id": "count"}, inplace=True)
    df2 = df2.merge(cnt)
    df3 = df2.copy()
    plt.figure(figsize=(15,10))
    for itm in items:
        df3[itm] = df3.apply(lambda x: x[itm]/x["count"], axis=1)
        p = sns.lineplot(data=df3, x="yearpublished", y=itm, label=itm, linewidth=1.8)
        p.set_xlabel("Publisehd year", fontsize=15)
        p.set_ylabel("Rate", fontsize=15)
        p.tick_params(labelsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=13)
    plt.title("Rate of Published Games per {} from {} to {}".format(group_name, lb, ub), fontsize=20)
    plt.show()


## Rating analysis
def summarize_per_attribute(df, threshold, lb, ub, items, popularity_metrics):
    """
    compute the average popular metrics value per group given items 

    df: dataframe
    threshold: the minimum number of ratings required to be included in mean computation
    lb: year lower bound
    ub: year upper bound
    items: group items to be considered
    popular metrics: list of popular metrics 

    """
    splits = {}; overallstats = {}
    filtered = df.loc[df["numratings"] >= threshold]
    filtered = filtered.loc[(df["yearpublished"] >= lb) & (df["yearpublished"] <= ub)]
    
    for cat in items:
        tmp = filtered.loc[filtered[cat] == 1, ["name", "yearpublished"] + popularity_metrics]
        splits[cat] = tmp.groupby("yearpublished").mean().reset_index()
        overallstats[cat] = tmp[popularity_metrics].mean()    
    return(splits, overallstats)

def plot_avg_stats(overallstats, metric, group_name):
    """
    display the barplot of averae stats per group

    overallstats: average popular metrics values by group
    metric: the name of the popular metric considered
    group_name: group name
    """
    y_vals = []
    for cat in overallstats.keys():
        y_vals.append(overallstats[cat][metric])
    zipped_list = list(zip(list(overallstats.keys()), y_vals))
    res = sorted(zipped_list, key = lambda x: -x[1]) 
    p = sns.barplot(x=[t[0] for t in res], y=[t[1] for t in res])
    p.set_xticklabels(p.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=14)
    p.tick_params(labelsize=15)
    p.set_title("Average {} per {}".format(metric, group_name), fontsize=16)


