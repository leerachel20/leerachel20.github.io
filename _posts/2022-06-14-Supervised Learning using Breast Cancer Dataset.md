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
![trainingtestingcomparison](/assets/images/image2_1.png)

The plot shows the training and test set accuracy on the y-axis against the setting of n_neighbors on the x-axis. While real-world plots are rarely very smooth, we can still recognize some of the characteristics of overfitting and underfitting.

Considering a single nearest neighbor, the prediction on the training set is perfect. But when more neighbors are considered, the model becomes simpler and the training accuracy drops.

The test set accuracy for using a single neighbor is lower than when using more neighbors, indicating that using the single nearest neighbor leads to a model that is too complex. On the other hand, when considering 10 neighbors, the model is too simple and performance is even worse. The best performance is somewhere in the middle, using around 6 neighbors. Still, it is good to keep the scale of the plot in mind. The worst performance is around 88% accuracy, which might still be acceptable.

In principle, there are two important parameters to the KNeighbors classifier: the number of neighbors and how you measure distance between data points. In practice, using a small number of neighbors like three or five often works well, but you should certainly adjust this parameter.

One of the **strengths** of k-NN is that the model is very easy to understand, and often gives reasonable performance without a lot of adjustments. Using this algorithm is a good baseline method to try before considering more advanced techniques. Building the nearest neighbors model is usually very fast, but when your training set is very large (either in number of features or in number of samples) prediction can be slow.

One of the **weaknesses** of k-NN is that When using the k-NN algorithm, it’s important to preprocess your data. This approach often does not perform well on datasets with many features (hundreds or more), and it does particularly badly with datasets where most features are 0 most of the time (so-called sparse datasets).

Therefore, while the nearest k-neighbors algorithm is easy to understand, ***it is not often used in practice***, due to prediction being slow and its inability to handle many features. The method we discuss next has neither of these drawbacks.
