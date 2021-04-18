#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import re


# def sub_feature_names(data, feature_names):
#     cols = data.columns.tolist()
#     feat_map = {"x" + str(num):cat for num, cat in enumerate(cols)}
#     feat_string = ",".join(feature_names)
#     feat_string
#     for key, value in feat_map.items():
#         feat_string = re.sub(fr"\b{key}\b", value, feat_string)
#     feat_string = feat_string.replace(" ", " : ").split(",")
#     return feat_string


def lr(X_train, X_test, y_train, y_test):
    # n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    # p = X_train.shape[1]
    
    # Tune hyperparameters
    Cs = np.logspace(-1, 5, 100)

    # Build model
    model = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear",
                                 random_state=42, n_jobs=3)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Get most important features
    # coefs = model.coef_.tolist()
    # coefs = [item for sublist in coefs for item in sublist]
    # feature_names = X_train.columns
    # coefficients = {}
    # for feature_name, coef in zip(feature_names, coefs):
    #     coefficients[feature_name] = coef
    # coefficients = {key: value for key, value in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    return model, train_pred, test_pred


def svm(X_train, X_test, y_train, y_test):
    n_folds = 5
    # param_grid = {"C": [0.1, 1, 10, 100, 1000]} 
    # model = GridSearchCV(LinearSVC(max_iter=10000, random_state=42), param_grid, cv=n_folds)
    model = LinearSVC(C=10000, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return model, train_pred, test_pred


def rf(X_train, X_test, y_train, y_test):
    # Tune hyperparameters
    Bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    Rsqs = []
    for B in Bs:
        model = RandomForestClassifier(n_estimators=B, max_depth=20, max_features="sqrt", random_state=42, n_jobs=3)
        model.fit(X_train, y_train)
        Rsqs.append(model.score(X_train, y_train))

    max_Rsq = max(Rsqs)
    max_index = Rsqs.index(max_Rsq)
    max_index
    B = Bs[max_index]
    
    # Build model
    model = RandomForestClassifier(n_estimators=B, max_depth=20, max_features="sqrt", oob_score=True, random_state=42, n_jobs=3)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    oob_score = model.oob_score_
    
    # Get feature importance
    feature_list = list(X_test.columns)
    importances = list(model.feature_importances_)
    feature_importances_list = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances_list = sorted(feature_importances_list, key=lambda x: x[1], reverse=True)
    feature_importances = {}
    for item in feature_importances_list:
        feature_importances[item[0]] = item[1]

    print("Number of trees: {}".format(B))
    print("Out-of-bag score: {}".format(oob_score))
    return model, train_pred, test_pred, oob_score, feature_importances


def gb(X_train, X_test, y_train, y_test):
    # Tune hyperparameters
    Bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    Rsqs = []
    for B in Bs:
        model = GradientBoostingClassifier(n_estimators=B, max_depth=20, max_features="sqrt", validation_fraction=0.20, random_state=42)
        model.fit(X_train, y_train)
        Rsqs.append(model.score(X_train, y_train))

    max_Rsq = max(Rsqs)
    max_index = Rsqs.index(max_Rsq)
    max_index
    B = Bs[max_index]

    #Build model
    model = GradientBoostingClassifier(n_estimators=B, max_depth=20, max_features="sqrt", validation_fraction=0.20, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Get feauture importance
    feature_list = list(X_test.columns)
    importances = list(model.feature_importances_)
    feature_importances_list = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances_list = sorted(feature_importances_list, key=lambda x: x[1], reverse=True)
    feature_importances = {}
    for item in feature_importances_list:
        feature_importances[item[0]] = item[1]

    print("Number of trees: {}".format(B))
    return model, train_pred, test_pred, feature_importances


def nb(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return model, train_pred, test_pred