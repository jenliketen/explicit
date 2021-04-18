#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import setp
from wordcloud import WordCloud
from collections import Counter
from nltk import FreqDist
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from gensim.corpora import Dictionary
# from gensim.models import LdaMulticore
# import pyLDAvis
# from pyLDAvis import gensim_models
from textblob import TextBlob


def boxplot_word_count(df):
    def setBoxColors(bp, c):
        setp(bp["boxes"][0], color=c)
        setp(bp["medians"][0], color=c)
        setp(bp["whiskers"][0], color=c)
        setp(bp["whiskers"][1], color=c)
        setp(bp["caps"][0], color=c)
        setp(bp["caps"][1], color=c)

    df["word_count"] = df["lemmatized"].apply(len)
    df["unique_word_count"] = df["unique_words"].apply(len)

    raw_explicit = df[df["explicit_label"] == 1]["word_count"]
    raw_clean = df[df["explicit_label"] == 0]["word_count"]
    unique_explicit = df[df["explicit_label"] == 1]["unique_word_count"]
    unique_clean = df[df["explicit_label"] == 0]["unique_word_count"]
    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()

    flierprops = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": "slateblue"}
    meanprops = {"marker": "v", "markerfacecolor": "slateblue", "markeredgecolor": "slateblue"}
    bp = plt.boxplot(raw_explicit, positions=[1.1], widths=0.3, flierprops=flierprops, meanprops=meanprops, showmeans=True)
    setBoxColors(bp, "slateblue")
    bp = plt.boxplot(unique_explicit, positions=[3.1], widths=0.3, flierprops=flierprops, meanprops=meanprops, showmeans=True)
    setBoxColors(bp, "slateblue")

    flierprops = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": "violet"}
    meanprops = {"marker": "v", "markerfacecolor": "violet", "markeredgecolor": "violet"}
    bp = plt.boxplot(raw_clean, positions=[1.5], widths=0.3, flierprops=flierprops, meanprops=meanprops, showmeans=True)
    setBoxColors(bp, "violet")
    bp = plt.boxplot(unique_clean, positions=[3.5], widths=0.3, flierprops=flierprops, meanprops=meanprops, showmeans=True)
    setBoxColors(bp, "violet")

    plt.xlim(0, 4.6)
    ax.set_xticklabels(["Raw", "Unique"])
    ax.set_xticks([1.3, 3.3])
    ax.set_ylabel("Number of words")
    ax.set_title("Word count per song")

    hB, = plt.plot([1, 1], "slateblue")
    hR, = plt.plot([1, 1], "violet")
    plt.legend((hB, hR),("Explicit", "Non-explicit"))
    hB.set_visible(False)
    hR.set_visible(False)


def hist_word_count(df):
    df["word_count"] = df["lemmatized"].apply(len)
    df["unique_word_count"] = df["unique_words"].apply(len)

    raw_explicit = df[df["explicit_label"] == 1]["word_count"]
    raw_clean = df[df["explicit_label"] == 0]["word_count"]
    unique_explicit = df[df["explicit_label"] == 1]["unique_word_count"]
    unique_clean = df[df["explicit_label"] == 0]["unique_word_count"]

    bins = np.linspace(0, 550, 10)

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(raw_explicit, bins, alpha=0.5, label="Explicit", color="slateblue", edgecolor="dimgray")
    plt.hist(raw_clean, bins, alpha=0.5, label="Non-explicit", color="violet", edgecolor="dimgray")
    plt.yscale("log", nonposy="clip")
    plt.xlabel("Number of words")
    plt.ylabel(r"$log$(Frequency)")
    plt.title("Distribution of raw word count per song")
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.hist(unique_explicit, bins, alpha=0.5, label="Explicit", color="slateblue", edgecolor="dimgray")
    plt.hist(unique_clean, bins, alpha=0.5, label="Non-explicit", color="violet", edgecolor="dimgray")
    plt.yscale("log", nonposy="clip")
    plt.xlabel("Number of words")
    plt.ylabel(r"$log$(Frequency)")
    plt.title("Distribution of unique word count per song")
    plt.legend(loc="upper right")


def plot_wordcloud(df):
    def get_words(df, explicit=True):
        words = df[df["explicit_label"] == 1]["lemmatized"]
        if explicit == False:
            words = df[df["explicit_label"] == 0]["lemmatized"]

        word_list = []
        for word in words:
            word_list += word

        return word_list
    
    fig = plt.figure(figsize=(30, 31), facecolor="white")
    
    plt.subplot(2, 1, 1)
    words = get_words(df, explicit=False)
    text = " ".join(words)
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Word frequency in non-explicit songs", fontsize=40)
    plt.axis("off")
    
    plt.subplot(2, 1, 2)
    words = get_words(df, explicit=True)
    text = " ".join(words)
    wordcloud = WordCloud(background_color="black").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Word frequency in explicit songs", fontsize=40)
    plt.axis("off")
    plt.tight_layout(pad=1)


def get_freq(df):
    df["lemma_str"] = [" ".join(map(str, l)) for l in df["lemmatized"]]
    grouped_words = df.groupby("explicit_label")["lemma_str"].apply(lambda x: Counter(" ".join(x).split()).most_common(25))

    clean = grouped_words.iloc[0]
    words_clean = list(zip(*clean))[0]
    freq_clean = list(zip(*clean))[1]

    explicit = grouped_words.iloc[1]
    words_explicit = list(zip(*explicit))[0]
    freq_explicit = list(zip(*explicit))[1]

    plt.figure(figsize=(15, 16))
    plt.subplot(2, 1, 1)
    plt.bar(words_clean, freq_clean, color="violet")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.yticks()
    plt.xticks(rotation=60)
    plt.title("Frequency of the 25 most common words for non-explicit songs")

    plt.subplot(2, 1, 2)
    p = plt.bar(words_explicit, freq_explicit, color="slateblue")
    
    # Highlight expletives
    p[3].set_color("crimson")
    p[4].set_color("crimson")
    p[10].set_color("crimson")
    p[11].set_color("crimson")
    
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.yticks()
    plt.xticks(rotation=60)
    plt.title("Frequency of the 25 most common words for explicit songs")
    plt.tight_layout(pad=1)


# def make_dtm(df):
#     tf_vectorizer = CountVectorizer(max_df=0.9, min_df=25, max_features=5000)
#     tf = tf_vectorizer.fit_transform(df["lemma_str"].values.astype("U"))
#     tf_feature_names = tf_vectorizer.get_feature_names()
#     dtm = pd.DataFrame(tf.toarray(), columns=list(tf_feature_names))
    
#     return dtm


# def lda(df):
#     corpus = df["unique_words"]
#     corpus_new = []
#     for i in corpus:
#         bad = ["go", "know", "make", "see", "come", "say",
#                "want", "time", "feel", "take", "look", "one"]
#         new = [x for x in i if x not in bad]
#         corpus_new.append(new)
#     dic = Dictionary(corpus_new)
#     bow_corpus = [dic.doc2bow(doc) for doc in corpus_new]
#     lda_model = LdaMulticore(bow_corpus,
#                              num_topics=4,
#                              id2word=dic,
#                              random_state=42,
#                              passes=2,
#                              workers=3)

#     pyLDAvis.enable_notebook()
#     vis = gensim_models.prepare(lda_model, bow_corpus, dic)
    
#     return lda_model, vis


def hist_sentiment(df):
    df["sentiment"] = df["lemma_str"].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 5))
    plt.hist(df["sentiment"], bins=50, color="darkorchid")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of songs")
    plt.title("Sentiment distribution")


def boxplot_sentiment(df):
    def setBoxColors(bp, c):
        setp(bp["boxes"][0], color=c)
        setp(bp["medians"][0], color=c)
        setp(bp["whiskers"][0], color=c)
        setp(bp["whiskers"][1], color=c)
        setp(bp["caps"][0], color=c)
        setp(bp["caps"][1], color=c)

    sent_explicit = df[df["explicit_label"] == 1]["sentiment"]
    sent_clean = df[df["explicit_label"] == 0]["sentiment"]
    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()

    flierprops = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": "slateblue"}
    meanprops = {"marker": "v", "markerfacecolor": "slateblue", "markeredgecolor": "slateblue"}
    bp = plt.boxplot(sent_explicit, positions=[1.1], widths=0.3, flierprops=flierprops, meanprops=meanprops, showmeans=True)
    setBoxColors(bp, "slateblue")
    
    flierprops = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": "violet"}
    meanprops = {"marker": "v", "markerfacecolor": "violet", "markeredgecolor": "violet"}
    bp = plt.boxplot(sent_clean, positions=[1.7], widths=0.3, flierprops=flierprops, meanprops=meanprops, showmeans=True)
    setBoxColors(bp, "violet")


    plt.xlim(0, 3.0)
    ax.set_xticks([])
    ax.set_ylabel("Sentiment")
    ax.set_title("Sentiment of songs")

    hB, = plt.plot([1, 1], "slateblue")
    hR, = plt.plot([1, 1], "violet")
    plt.legend((hB, hR),("Explicit", "Non-explicit"))
    hB.set_visible(False)
    hR.set_visible(False)