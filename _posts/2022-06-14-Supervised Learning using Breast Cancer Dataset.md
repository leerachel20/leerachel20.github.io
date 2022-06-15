---
title: "Supervised Learning Model on Breast Cancer Dataset"
mathjax: true
layout: post
categories:
  - github
  - website
---

# Supervised Learning
Supervised learning is used whenever we want to predict a certain
outcome from a given input, and we have examples of input/output pairs. We build a
machine learning model from these input/output pairs, which comprise our training
set. Our goal is to make accurate predictions for new, never-before-seen data. Supervised
learning often requires human effort to build the training set, but afterward
automates and often speeds up an otherwise laborious or infeasible task.

## Classification and Regression
There are two major types of supervised machine learning problems, called *classification*
and *regression*.

In **classification**, the goal is to predict a class label, which is a choice from a predefined
list of possibilities. You can think of binary classification as trying to answer a yes/no question. 
Classifying emails as either spam or not spam is an example of a binary classification problem.

For **regression** tasks, the goal is to predict a continuous number, or a floating-point
number in programming terms (or real number in mathematical terms). Predicting a
person’s annual income from their education, their age, and where they live is an
example of a regression task. When predicting income, the predicted value is an
amount, and can be any number in a given range.

```python
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
```
```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
```
```md
cancer.keys(): 
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```
Of these 569 data points, 212 are labeled as malignant and 357 as benign.

## Model complexity and generalization: Analyzing the KNeighborsClassifier
We evaluate training and test set performance with different numbers of neighbors.
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 1000)

training_accuracy = []
test_accuracy = []
neighbors = range(1,11)

for n_neighbors in neighbors:
  kNN = KNeighborsClassifier(n_neighbors = n_neighbors)
  kNN.fit(X_train, y_train)
  training_accuracy.append(kNN.score(X_train, y_train))
  test_accuracy.append(kNN.score(X_test, y_test))

plt.plot(neighbors, training_accuracy, label = 'Training Accuracy')
plt.plot(neighbors, test_accuracy, label = 'Test Accuracy')
plt.ylabel('Accuracay')
plt.xlabel('n_neighbors')
plt.title('Comparison of training and test accuracy as a function of n_neighbors')
plt.legend()
```
