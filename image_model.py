import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------- PATH ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

IMG_SIZE = 64

X = []
y = []
labels = ["unripe", "mature", "ripe"]   # folder names

# ---------- LOAD IMAGES ----------
for idx, label in enumerate(labels):
    folder_path = os.path.join(IMAGE_DIR, label)

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()

            X.append(img)
            y.append(idx)

X = np.array(X)
y = np.array(y)

# ---------- TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------- MODEL ----------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------- PREDICT ----------
y_pred = model.predict(X_test)

# ---------- RESULTS ----------
print("📸 IMAGE MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=labels))
