import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("E:\Research\PlayerBehaviour\dataset\PlayerStats.csv")

X = data.drop(columns=["uuid", "playerName", "label"])

y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

plt.figure(figsize=(80,40))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
