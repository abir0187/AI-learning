# Load Algorithm
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()


features = iris['data']
label = iris['target']

algorithm = tree.DecisionTreeClassifier()
algorithm.fit(features, label)


print(algorithm.predict([[6.7, 3.1, 5.6, 2.4]]))