import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
print(f"\nAUC (Macro OVR): {auc:.4f}")

# Note: TP, TN, FP, FN are typically calculated per class from the confusion matrix.
# Example for class 0 (setosa) vs rest:
# TP = cm[0, 0]
# FN = cm[0, 1] + cm[0, 2]
# FP = cm[1, 0] + cm[2, 0]
# TN = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]
# Recall = TP / (TP + FN)
# Specificity = TN / (TN + FP)
# F1-Score = classification_report provides this per class.

