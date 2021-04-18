#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_roc_curve


def make_cm(y_true, y_pred):
	cm_values = confusion_matrix(y_true, y_pred)
	cm = pd.DataFrame({"actual_clean": cm_values[:, 0],
	                   "actual_explicit": cm_values[:, 1]},
	                   index=["predicted_clean", "predicted_explicit"])

	return cm


def make_summary(y_train, y_test, train, test):
	def pr(y_true, y_pred):
		
		precision = precision_score(y_true, y_pred, average="binary")
		recall = recall_score(y_true, y_pred, average="binary")
		f1 = f1_score(y_true, y_pred, average="binary")

		return precision, recall, f1

	precision_train = []
	precision_test = []
	recall_train = []
	recall_test = []
	f1_train = []
	f1_test = []

	for pred in train:
		p, r, f = pr(y_train, pred)
		precision_train.append(p)
		recall_train.append(r)
		f1_train.append(f)

	for pred in test:
		p, r, f = pr(y_test, pred)
		precision_test.append(p)
		recall_test.append(r)
		f1_test.append(f)

	summary = pd.DataFrame({
		"precision_test": precision_test,
		"recall_test": recall_test,
		"f1_test": f1_test,
		"precision_train": precision_train,
		"recall_train": recall_train,
		"f1_train": f1_train},
		index=[
		"logistic_regression", "linear_svm",
		"random_forest"," gradient_boost",
		"naive_bayes"])

	return summary


def get_roc_auc(models, names, X_train, X_test, y_train, y_test):
	auc_test = []
	fig = plt.figure(figsize=(10, 5))
	ax1 = fig.add_subplot(1, 2, 1)
	for model in zip(models, names):
	    roc_test = plot_roc_curve(model[0], X_test, y_test, ax=ax1, name=model[1])
	    auc_test.append(roc_test.roc_auc)
	ax1.plot([0, 1], [0, 1], "k--")
	ax1.set_title("Test ROC")

	auc_train = []
	ax2 = fig.add_subplot(1, 2, 2)
	for model in zip(models, names):
	    roc_train = plot_roc_curve(model[0], X_train, y_train, ax=ax2, name=model[1])
	    auc_train.append(roc_train.roc_auc)
	ax2.plot([0, 1], [0, 1], "k--")
	ax2.set_title("Training ROC")