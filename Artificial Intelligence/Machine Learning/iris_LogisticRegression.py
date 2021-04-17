from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = load_iris()
iris_target = data.target
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
iris_all = iris_features.copy()
iris_all['target'] = iris_target
# sns.pairplot(data=iris_all,diag_kind='hist',hue='target')
# plt.show()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
iris_all_class0 = iris_all[iris_all['target'] == 0].values
iris_all_class1 = iris_all[iris_all['target'] == 1].values
iris_all_class2 = iris_all[iris_all['target'] == 2].values
ax.scatter(iris_all_class0[:, 0], iris_all_class0[:,
           1], iris_all_class0[:, 2], label='setosa')
ax.scatter(iris_all_class1[:, 0], iris_all_class1[:, 1],
           iris_all_class1[:, 2], label='versicolor')
ax.scatter(iris_all_class2[:, 0], iris_all_class2[:, 1],
           iris_all_class2[:, 2], label='birginica')
# plt.legend()
# plt.show()
iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]


def TwoSplit(iris_features, iris_target):
    x_train, x_test, y_train, y_test = train_test_split(
        iris_features_part, iris_target_part, test_size=0.2, random_state=2020)
    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf.fit(x_train, y_train)
    predictdata(x_train, x_test, y_train, y_test,clf)


def ThreeSplit(iris_features, iris_target):
    x_train, x_test, y_train, y_test = train_test_split(
        iris_features, iris_target, test_size=0.2, random_state=2020)
    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf.fit(x_train, y_train)
    predictdata(x_train, x_test, y_train, y_test,clf)

def predictdata(x_train, x_test, y_train, y_test,clf):
    print("w & b:", clf.coef_, clf.intercept_)
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)
    print("train accuracy:", metrics.accuracy_score(y_train, train_predict))
    print("test accuracy:", metrics.accuracy_score(y_test, test_predict))
    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print("confusion matrix:", confusion_matrix_result)
    plt.figure(figsize=(8, 5))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Reds')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True labels')
    plt.show()
#TwoSplit(iris_features,iris_target)
ThreeSplit(iris_features,iris_target)