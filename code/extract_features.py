import os
import librosa
import numpy as np
import pandas as pd

# -------- PATHS --------
DATA_DIR = r"C:\Users\BASAVARAJ PATIL\PycharmProjects\pythonProject4\data\raw"
OUTPUT_CSV = r"C:\Users\BASAVARAJ PATIL\PycharmProjects\pythonProject4\data\features.csv"

# -------- FEATURE EXTRACTION --------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc, spectral_centroid, zero_crossing])

# -------- MAIN --------
rows = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(label_path, file)
            features = extract_features(file_path)
            rows.append(list(features) + [label])

columns = [f"mfcc_{i}" for i in range(13)] + ["spectral_centroid", "zero_crossing", "label"]

df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Feature extraction completed successfully!")
print("📁 CSV saved at:", OUTPUT_CSV)

