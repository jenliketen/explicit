#!/usr/bin/env python3

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def vectorize(df):
    tf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=25, max_features=5000)
    tf = tf_vectorizer.fit_transform(df["lemma_str"].values.astype("U"))
    tf_feature_names = tf_vectorizer.get_feature_names()
    vec = pd.DataFrame(tf.toarray(), columns=list(tf_feature_names))

    df_vec = pd.concat([df[["word_count","unique_word_count",
    	"sentiment", "explicit_label"]].reset_index(drop=True),
    	vec], axis=1)
    
    return df_vec


def resample_data(df, direction="up"):
    majority = df[df["explicit_label"] == 0]
    minority = df[df["explicit_label"] == 1]
    
    # Upsample minority class
    minority_upsampled = resample(minority, replace=True,
                                  n_samples=len(majority),
                                  random_state=42)

    df_resampled = pd.concat([majority, minority_upsampled]).reset_index(drop=True)
    
    # Downsample majority class
    if direction == "down":
        majority_downsampled = resample(majority, replace=False,
                                        n_samples=len(minority),
                                        random_state=42)

        df_resampled = pd.concat([majority_downsampled, minority]).reset_index(drop=True)
    
    return df_resampled



def split(df, standardize=False):
    X = df.drop("explicit_label", axis=1, inplace=False)
    y = df["explicit_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if standardize == True:
        X_train[["word_count", "unique_word_count"]] -= X_train[["word_count", "unique_word_count"]].mean(axis=0)
        X_train[["word_count", "unique_word_count"]] /= X_train[["word_count", "unique_word_count"]].std(axis=0)
        X_test[["word_count", "unique_word_count"]] -= X_test[["word_count", "unique_word_count"]].mean(axis=0)
        X_test[["word_count", "unique_word_count"]] /= X_test[["word_count", "unique_word_count"]].std(axis=0)
    
    return X_train, X_test, y_train, y_test