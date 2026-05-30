import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

print("Dataset shape:", data.shape)  # Should be (N, 42)

# Split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {score*100:.2f}%")

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✅ Model saved")