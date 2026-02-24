import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# =========================
# PATHS
# =========================
AUDIO_CSV = r"C:\Users\BASAVARAJ PATIL\PycharmProjects\pythonProject4\data\features.csv"
IMAGE_DIR = r"C:\Users\BASAVARAJ PATIL\PycharmProjects\pythonProject4\data\images"

IMG_SIZE = (64, 64)

# =========================
# LOAD AUDIO FEATURES
# =========================
audio_df = pd.read_csv(AUDIO_CSV)

X_audio = audio_df.drop("label", axis=1)
y_audio = audio_df["label"]

# Encode labels
le = LabelEncoder()
y_audio_enc = le.fit_transform(y_audio)

# =========================
# LOAD IMAGE FEATURES
# =========================
image_features = []
image_labels = []

for label in os.listdir(IMAGE_DIR):
    label_path = os.path.join(IMAGE_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(label_path, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = img.flatten() / 255.0  # normalize

            image_features.append(img)
            image_labels.append(label)

X_img = np.array(image_features)
y_img_enc = le.transform(image_labels)

# =========================
# ALIGN AUDIO + IMAGE SIZE
# =========================
min_samples = min(len(X_audio), len(X_img))

X_audio = X_audio.iloc[:min_samples].values
X_img = X_img[:min_samples]
y = y_audio_enc[:min_samples]

# =========================
# FEATURE FUSION
# =========================
X_fused = np.hstack((X_audio, X_img))

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y, test_size=0.3, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =========================
# RESULTS
# =========================
y_pred = model.predict(X_test)

print("\n🔗 FUSION MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
import pickle

# Save trained model to file
with open("coconut_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model successfully saved as 'coconut_model.pkl'")
import pickle

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
