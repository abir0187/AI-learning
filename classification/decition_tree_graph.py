from sklearn import tree
from sklearn.datasets import load_iris
import pydotplus
from sklearn.externals.six import StringIO

# 1. load dataset and train classifier
iris = load_iris()
classifier = tree.DecisionTreeClassifier()
classifier.fit(iris.data, iris.target)

# 2. visualize
dot_data = StringIO()
tree.export_graphviz(classifier, out_file=dot_data,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")