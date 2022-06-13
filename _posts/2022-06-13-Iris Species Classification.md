---
title: "Iris Species Classification"
mathjax: true
layout: post
categories:
  - github
  - website
---

![Iris](/assets/iris.jpg)

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
```
