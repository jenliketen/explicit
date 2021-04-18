# NLP: CLASSIFICATION OF SONG LYRICS WITH EXPLICIT CONTENT

## Introduction

With the migration of music from physical records to online, we need to find a way to accurately label explicit songs. This can be achieved through building a binary classifier. A successful classifier will have many real-life impacts. It will allow music streaming services (_eg._ Spotify, Apple Music) to label mass volumes of songs that come in on a daily basis. It will make it easy for parents to safeguard playlists before their children (especially grade school children) have access to them. Furthermore, it will help composers, lyricists, and other musicians to test out their music and make sure it reaches their target audience before release. This project aims to build a classifier that does exactly this.

## How to use

### Data

All data, raw or processed, are stored in the `data` folder. To save space, we only upload the first few rows of each dataset.

### Source code

All source code scripts are stored in the `src` folder and executable from the command line. They contain functions called by the code in the Jupyter Notebooks.

### Jupyter Notebooks
The Jupyter Notebooks are numbered in order of execution. You can just read through them in the order presented!

## Models

We have built the following models:
* Logistic regression with L1 penalty
* Linear support vector machine (SVM)
* Random forest
* Gradient boost
* Na&iuml;ve Bayes

## Results

Our best model is the gradient boost. Here are the test set performance metrics:
* FNR: 0.015
* FPR: 0
* Precision: 0.985
* Recall: 1.000
* F1 score: 0.992