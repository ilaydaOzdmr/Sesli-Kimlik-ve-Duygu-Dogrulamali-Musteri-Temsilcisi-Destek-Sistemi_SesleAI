import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# 1. Parametreler
# -----------------------------
DATA_PATH =r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\Audio_Speech_Actors_01-24 (1)"

MODEL_PATH =r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\cnn_gru_final_model.h5"
n_mfcc = 40

# -----------------------------
# 2. Veri Hazırlama
# -----------------------------
X, y = [], []
for label in os.listdir(DATA_PATH):
    class_path = os.path.join(DATA_PATH, label)
    if not os.path.isdir(class_path):
        continue
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        try:
            signal, sr = librosa.load(file_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T
            X.append(mfcc)
            y.append(label)
        except Exception as e:
            print(f"Hata: {file_path} -> {e}")

# Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = max([mfcc.shape[0] for mfcc in X])
X_pad = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# Normalizasyon
from sklearn.preprocessing import StandardScaler
nsamples, nx, ny = X_pad.shape
X_pad = X_pad.reshape((nsamples * nx, ny))
scaler = StandardScaler()
X_pad = scaler.fit_transform(X_pad)
X_pad = X_pad.reshape((nsamples, nx, ny))

# Label encode
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/Test split (ayrı test için)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# -----------------------------
# 3. Model Yükle
# -----------------------------
model = load_model(MODEL_PATH)
print(model.summary())

# -----------------------------
# 4. Test Et
# -----------------------------
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Accuracy
acc = np.mean(y_pred_labels == y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_labels, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - CNN+GRU")
plt.show()

# -----------------------------
# 5. Eğitim Grafikleri (history varsa)
# -----------------------------
try:
    import pickle
    with open("cnn_gru_history.pkl", "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="Train Acc")
    plt.plot(history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
except:
    print("⚠️ History dosyası bulunamadı, sadece test sonuçları çizildi.")
