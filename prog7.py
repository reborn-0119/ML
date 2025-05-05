import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, alpha=0.0001, solver='adam')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
y_prob = mlp.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=cancer.target_names)
auc = roc_auc_score(y_test, y_prob)

tn, fp, fn, tp = cm.ravel()
error_rate = (fp + fn) / (tn + fp + fn + tp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)


print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print("\nConfusion Matrix:\n", cm)
print(f"\nTrue Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"\nRecall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

print(f"AUC: {auc:.4f}")
print("\nClassification Report:\n", report)