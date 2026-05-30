import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
#                    LOAD DATASET
# ==========================================================

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

print("Dataset Shape:", data.shape)

# ==========================================================
#                 TRAIN TEST SPLIT
# ==========================================================

x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=42
)

# ==========================================================
#                     MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================================
#                     TRAIN MODEL
# ==========================================================

print("\nTraining Model...\n")

model.fit(x_train, y_train)

# ==========================================================
#                     PREDICTION
# ==========================================================

y_pred = model.predict(x_test)

# ==========================================================
#                     METRICS
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# ==========================================================
#                 CROSS VALIDATION
# ==========================================================

cv_scores = cross_val_score(
    model,
    data,
    labels,
    cv=5
)

# ==========================================================
#                 PRINT RESULTS
# ==========================================================

print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

print(f"\n✅ Precision: {precision * 100:.2f}%")

print(f"\n✅ Recall: {recall * 100:.2f}%")

print(f"\n✅ F1-Score: {f1 * 100:.2f}%")

print("\n✅ Cross Validation Scores:")
print(cv_scores)

print(f"\n✅ Average CV Accuracy: {cv_scores.mean() * 100:.2f}%")

# ==========================================================
#               CLASSIFICATION REPORT
# ==========================================================

print("\n📋 Classification Report:\n")

report = classification_report(y_test, y_pred)

print(report)

# ==========================================================
#               CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

# ==========================================================
#                   SAVE MODEL
# ==========================================================

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("\n✅ Model Saved Successfully")

# ==========================================================
#                    VISUALIZATION
# ==========================================================

# ==========================================================
#               ACCURACY LINE PLOT
# ==========================================================

plt.figure(figsize=(8,5))

accuracy_values = [
    accuracy * 100,
    cv_scores.mean() * 100
]

labels_graph = [
    'Test Accuracy',
    'Average CV Accuracy'
]

x = np.arange(len(labels_graph))

plt.plot(
    x,
    accuracy_values,
    marker='o',
    linewidth=3,
    markersize=10,
    label='Model Accuracy'
)

plt.xticks(x, labels_graph)

plt.ylabel("Accuracy (%)")

plt.title("Model Accuracy Comparison")

# Value labels
for i, value in enumerate(accuracy_values):

    plt.text(
        x[i],
        value + 0.5,
        f"{value:.2f}%",
        ha='center',
        fontsize=11
    )

plt.grid(True)

plt.legend()

plt.show()

# ==========================================================
#           CROSS VALIDATION ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

folds = np.arange(1, len(cv_scores)+1)

plt.plot(
    folds,
    cv_scores * 100,
    marker='o',
    linewidth=2,
    markersize=8,
    label='CV Accuracy'
)

plt.xlabel("Fold")

plt.ylabel("Accuracy (%)")

plt.title("Cross Validation Accuracy")

# Value labels
for i, score in enumerate(cv_scores):

    plt.text(
        folds[i],
        score * 100 + 0.5,
        f"{score*100:.1f}%",
        ha='center'
    )

plt.grid(True)

plt.legend()

plt.show()

# ==========================================================
#               CONFUSION MATRIX HEATMAP
# ==========================================================

plt.figure(figsize=(14,10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.unique(labels),
    yticklabels=np.unique(labels)
)

plt.xlabel("Predicted Labels")

plt.ylabel("True Labels")

plt.title("Confusion Matrix")

# ==========================================================
#                METRICS LEGEND BOX
# ==========================================================

metrics_text = (
    f"Accuracy  : {accuracy*100:.2f}%\n"
    f"Precision : {precision*100:.2f}%\n"
    f"Recall    : {recall*100:.2f}%\n"
    f"F1-Score  : {f1*100:.2f}%"
)

plt.gcf().text(
    0.92,
    0.5,
    metrics_text,
    fontsize=12,
    bbox=dict(
        facecolor='white',
        edgecolor='black'
    )
)

plt.show()