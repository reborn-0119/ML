import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from itertools import cycle

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Binarize the output for multi-class ROC (One-vs-Rest)
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# Split data
X_train, X_test, y_train, y_test_bin = train_test_split(X_scaled, y_bin, test_size=0.3, random_state=42)
_, _, y_train_orig, y_test_orig = train_test_split(X_scaled, y, test_size=0.3, random_state=42) # Keep original labels for report

# Train model
# Using probability=True for predict_proba, needed for ROC
svm_clf = SVC(kernel='linear', probability=True, random_state=42)
# Fit on original multi-class labels
svm_clf.fit(X_train, y_train_orig)
y_pred = svm_clf.predict(X_test)
y_prob = svm_clf.predict_proba(X_test)

# Evaluate (using original multi-class labels for standard metrics)
accuracy = accuracy_score(y_test_orig, y_pred)
cm = confusion_matrix(y_test_orig, y_pred)
report = classification_report(y_test_orig, y_pred, target_names=iris.target_names)

# Calculate AUC (Macro OVR)
auc_macro = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
print(f"\nAUC (Macro OVR): {auc_macro:.4f}")

# Plotting ROC Curve for each class (One-vs-Rest)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_prob[:, i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc["micro"] = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='micro')


plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(iris.target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - SVM (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()