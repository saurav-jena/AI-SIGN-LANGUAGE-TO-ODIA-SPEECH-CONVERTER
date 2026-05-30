import os
import pickle
import mediapipe as mp
import cv2

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = "./data"

data = []
labels = []

for label in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            # 👉 ONLY FIRST HAND
            hand_landmarks = results.multi_hand_landmarks[0]

            x_ = []
            y_ = []
            data_aux = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Ensure correct size (21 landmarks × 2 = 42)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(label)

# Save dataset
with open("data.pickle", "wb") as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Dataset created successfully")
print("Samples:", len(data))