import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from transformer import TransformerBlock
  # transformer.py içindeyse böyle çağırabilirsin

# -----------------------------
# 1. Parametreler
# -----------------------------
DATA_PATH =r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\Audio_Speech_Actors_01-24 (1)"
n_mfcc = 40

# -----------------------------
# 2. Veri yükleme ve MFCC
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

# -----------------------------
# 3. Padding
# -----------------------------
max_len = max([mfcc.shape[0] for mfcc in X])
X_pad = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# -----------------------------
# 4. Normalizasyon
# -----------------------------
nsamples, nx, ny = X_pad.shape
X_pad = X_pad.reshape((nsamples * nx, ny))
scaler = StandardScaler()
X_pad = scaler.fit_transform(X_pad)
X_pad = X_pad.reshape((nsamples, nx, ny))

# -----------------------------
# 5. Label encode
# -----------------------------
le = LabelEncoder()
y_enc = to_categorical(le.fit_transform(y))

# -----------------------------
# 6. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# -----------------------------
# 7. Modelleri test et
# -----------------------------
model_files = [
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\cnn_gru_final_model.h5",
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\gru_model.h5",
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\lstm_model.h5",
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\rnn_model_final.h5",
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\transformer_model.h5",
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\wav2vec_model_fixed.h5",
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\cnn_lstm_model.h5",
    r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\CNN_model.h5"
]

results = {}

for model_file in model_files:
    if not os.path.exists(model_file):
        print(f"Model bulunamadı: {model_file}")
        continue

    print(f"\n--- {model_file} test ediliyor ---")

    try:
        if "transformer_model" in model_file:
            model = load_model(model_file, custom_objects={"TransformerBlock": TransformerBlock})
        else:
            model = load_model(model_file)

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results[model_file] = (acc, loss)
        print(f"{model_file} -> Accuracy: {acc:.4f}, Loss: {loss:.4f}")
    except Exception as e:
        print(f"{model_file} yüklenemedi! Hata: {e}")

# -----------------------------
# 8. Sonuçları tablo + grafik
# -----------------------------
print("\n=== Model Karşılaştırma Sonuçları ===")
for model_name, (acc, loss) in results.items():
    print(f"{model_name}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")

# Bar chart
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [v[0] for v in results.values()])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Modellerin Test Accuracy Karşılaştırması")
plt.tight_layout()
plt.show()
