# Load Algorithm
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)


algorithm = tree.DecisionTreeClassifier()
algorithm.fit(x_train, y_train)

pred = (algorithm.predict(x_test))

print(accuracy_score(y_test, pred))