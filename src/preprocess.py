#!/usr/bin/env python3

import pandas as pd
import contractions
import emoji
import string
import re
from src.stopwords import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer


# Clean lyrics and tokenize
def prep_lyrics(df, path):
    def remove_emojis(text):
        return text.encode("ascii", "ignore").decode("ascii")
    def remove_punctuation(text):
        punct = []
        punct += list(string.punctuation)
        punct += '’'
        #punct.remove("'")
        for punctuation in punct:
            text = text.replace(punctuation, " ")
        return text
    
    # General preprocessing
    df["prepped_lyrics"] = df["text"].apply(lambda x: contractions.fix(x)) # Expand contractions
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.replace("\n", " "))
    df["prepped_lyrics"] = df["prepped_lyrics"].str.replace("http\S+|www.\S+", "", case=False) # Remove hyperlinks
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.replace("&gt;", "")) # Remove "&gt;"
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: remove_emojis(x))
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(remove_punctuation)
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: str(x).replace(" s ", " ")) # Remove "s" after removing possessive apostrophe
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: re.sub("\w*\d\w*", "", x)) # Remove digits
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.lower()) # Bring to lowercase
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: " ".join(x.split())) # Replace multiple spaces with a single space
    
    # Remove words used to separate lyric sections
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.replace("hook ", ""))
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.replace("chorus ", ""))
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.replace("verse ", ""))
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.replace("intro ", ""))
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: x.replace("outro ", ""))

    df = df.reset_index(drop=True)
    df.to_csv(path + "/data.csv", index=False, sep=",")


# Lemmatize
def lemmatize(df, path):
    def get_wordnet_pos(word):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # Tokenize and remove stopwords
    #stop_words = set(stopwords.words("english")) # List of stopwords
    #stop_words = stopwords
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(word_tokenize) # Tokenize
    df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: [word for word in x if not word in stopwords]) # Remove stopwords
    
    # Get part-of-speech for each word
    # df["prepped_lyrics"] = df["prepped_lyrics"].apply(pos_tag)
    # df["prepped_lyrics"] = df["prepped_lyrics"].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    
    # Lemmatize according to part-of-speech
    wnl = WordNetLemmatizer()
    df["lemmatized"] = df["prepped_lyrics"].apply(lambda x: [wnl.lemmatize(word, get_wordnet_pos(word)) for word in x])

    # Get unique words
    df["unique_words"] = df["prepped_lyrics"].apply(lambda x: [wnl.lemmatize(word, get_wordnet_pos(word)) for word in set(x)])

    df["lemmatized"] = df["lemmatized"].apply(lambda x: [word for word in x if not word in stopwords])
    df["unique_words"] = df["unique_words"].apply(lambda x: [word for word in x if not word in stopwords])
    
    df = df.reset_index(drop=True)
    df.to_csv(path + "/data.csv", index=False, sep=",")