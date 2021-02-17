from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from string import punctuation

def get_stop_words():
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend([c for c in punctuation] + ['quot', 'rsquo', 'mdash', 'ndash','s'])
    return set(stop_words)


def generate_word_cloud(ds, max_words=1000, width=500, height=500, background_color='white'):
    '''Generate a word cloud from a pandas data series with text data'''
    assert isinstance(ds, pd.Series)

    stop_words = get_stop_words()
    tokenized_ds = ds.dropna().apply(nltk.word_tokenize)
    
    words = [word.lower() for ls in tokenized_ds for word in ls if word not in stop_words]
    
    wc = WordCloud(background_color=background_color, max_words=max_words, width=width, height=height)
    wc.generate(' '.join(words))
    plt.imshow(wc)
    plt.axis('off')
    fig=plt.gcf()
    fig.set_size_inches(15,10)
    plt.show()