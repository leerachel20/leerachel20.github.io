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
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
```
