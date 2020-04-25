# Load Algorithm
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

algorithm = tree.DecisionTreeClassifier()
algorithm.fit(iris.data, iris.target)


print(algorithm.predict([[6.7, 3.1, 5.6, 2.4]]))