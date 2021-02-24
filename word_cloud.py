def plot_word_cloud(word_counts):
    '''
    generate word clouds for a list of words
    '''
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS

    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate_from_frequencies(word_counts)

    plt.figure(figsize=(20, 15))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off")


def processing_columns(df_cols):
    '''
    unpack the individual terms and count them.

    '''
    import itertools as it
    from collections import Counter

    return Counter(list(it.chain(*df_cols)))