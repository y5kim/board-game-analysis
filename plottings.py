import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_barplot(df, colname, bins):
    '''
    Takes in dataframe and the column name you want to plot barplot

    :param df(pandas dataframe)
    :param colname(str): the name of the column
    :param bins(list): bins size
    '''
    #create bin labels based on bin intervals
    labels=[]
    for i in range(len(bins)-1):
        labels.append(str(bins[i]) + '-'+str(bins[i+1]))
    out=df.groupby(pd.cut(df[colname], bins=bins, labels=labels)).size().reset_index(name='count')

    #plot bar plot within given columns with the given bin sizes
    plt.bar(out[colname], out['count'])
    plt.title('Bar plot for ' + colname)
    plt.xlabel(colname)
    plt.ylabel('count')



def correlation_plot(df):

    '''
    Draw correlation heat map

    :param df(pandas dataframe)
    '''

    #correlation matrix
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    #colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    #draw heatmap
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin = -1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    #tilt xlabels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30) 


def read_correlation(corr_dataframe):
    '''
    read off correlations from given two columns
    '''
    #ask for an input
    col1, col2 = input("Enter two colnames(seprating by space): ").split()

    return corr_dataframe[col1][col2]


    


