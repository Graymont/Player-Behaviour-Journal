import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  

data = pd.read_csv(r"E:\Research\PlayerBehaviour\dataset\PlayerStats.csv")

X = data.drop(columns=["uuid", "playerName", "label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

importances = rf_clf.feature_importances_
feat_importance = pd.Series(importances, index=X.columns)
feat_importance = feat_importance.sort_values(ascending=False)




plt.figure(figsize=(12,6))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

joblib.dump(rf_clf, 'player_behaviour_model.pkl')




print("✅ Model berhasil disimpan ke 'player_behaviour_model.pkl'")
