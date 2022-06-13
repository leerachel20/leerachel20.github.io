---
title: "Iris Species Classification"
mathjax: true
layout: post
categories:
  - github
  - website
---

![Iris](/assets/images/iris.jpg)

## Data Set Information ##

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Predicted attribute: class of iris plant.

This is an exceedingly simple domain.

Our goal is to build a machine learning model that can learn from the measurements of these irises whose species is known, so that we can predict the species for a new
iris.

Because we have measurements for which we know the correct species of iris, this is a supervised learning problem. In this problem, we want to predict one of several
options (the species of iris). This is an example of a classification problem. The possible outputs (different species of irises) are called classes. Every iris in the dataset belongs to one of three classes, so this problem is a three-class classification problem. The desired output for a single data point (an iris) is the species of this flower. For a particular data point, the species it belongs to is called its label.

### Loading the Data ###
The data we will use for this example is the Iris dataset, a classical dataset in machine learning and statistics. It is included in scikit-learn in the datasets module. We can load it by calling the load_iris function:

```python
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Keys of iris_dataset:\n", iris_dataset.keys())
```
```md
Keys of iris_dataset:
 dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```

```python
print("Target names:", iris_dataset['target_names'])
print("Feature names:\n", iris_dataset['feature_names'])
```
```md
Target names: ['setosa' 'versicolor' 'virginica']
Feature names:
 ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```

```python
print("Type of data:", type(iris_dataset['data']))

print("Shape of data:", iris_dataset['data'].shape)
print("Shape of target: {}".format(iris_dataset['target'].shape))
```
```md
Type of data: <class 'numpy.ndarray'>
Shape of data: (150, 4)
Shape of target: (150,)
```

We see that the array contains measurements for 150 different flowers. `target` is a one-dimensional array, with one entry per flower.

```python
print("Target:\n{}".format(iris_dataset['target']))
```
```md
Target:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
```
0 means setosa, 1 means versicolor, and 2 means virginica.


### Training and Testing Data
First, we need to assess the model performance. we show it new data (data that it hasn’t seen before) for which we have labels. This is usually done by splitting the labeled data we have collected into two parts. One part of the data is used to build our machine learning model, and is called the training data. The rest of the data will be used to assess how well the model works; this is called the test data.

The **`train_test_split`** function under **`scikit-learn`** will extract 75% of the rows in the data as the training set, together with the corresponding labels for this data. The remaining 25% of the data, together with the remaining labels, is declared as the test set.

In `scikit-learn`, data is usually denoted with a capital X (two-dimensional array), while labels are denoted by a lowercase y (one-dimensional array - a vector).

Here, I am splitting 20% of the data as training set, `test_size = 0.20`.

```python
from sklearn.model_selection import train_test_split
X = iris_dataset['data']
y = iris_dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
```
```md
X_train shape: (120, 4)
y_train shape: (120,)
X_test shape: (30, 4)
y_test shape: (30,)
```

### Data Inspection
With 4 features included in the dataset, we will use the *pair plot* to inspect the data through visualization. 
```python
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
print(y_train)
```
```md
[2 1 0 2 2 1 0 1 1 1 2 0 2 0 0 1 2 2 2 2 1 2 1 1 2 2 2 2 1 2 1 0 2 1 1 1 1
 2 0 0 2 1 0 0 1 0 2 1 0 1 2 1 0 2 2 2 2 0 0 2 2 0 2 0 2 2 0 0 2 0 0 0 1 2
 2 0 0 0 1 1 0 0 1 0 2 1 2 1 0 2 0 2 0 0 2 0 2 1 1 1 2 2 1 1 0 1 2 2 0 1 1
 1 1 0 0 0 2 1 2 0]
```
```python
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
```
![Iris](/assets/images/irispairplot.png)
