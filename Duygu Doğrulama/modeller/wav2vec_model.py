# wav2vec2_train.py

import os
import torch
import torchaudio
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# === Parametreler ===
DATA_PATH = r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\Audio_Speech_Actors_01-24 (1)"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000  # Wav2Vec2 için zorunlu

# === Emotion mapping (RAVDESS) ===
emotion_map = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}


# === Model & Processor ===
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)

X, y = [], []

# === Veri yükleme ===
for actor in os.listdir(DATA_PATH):
    class_path = os.path.join(DATA_PATH, actor)
    if not os.path.isdir(class_path):
        continue
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        try:
            # Ses yükle
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.mean(dim=0)  # mono

            # Resample
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)

            # Wav2Vec2 giriş
            inputs = processor(waveform, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = wav2vec_model(inputs.input_values.to(DEVICE))
                hidden_states = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            X.append(hidden_states.squeeze())

            # === Etiket çıkarma (03-01-05-01-01-01-01.wav) ===
            emotion_id = int(file.split("-")[2])
            if emotion_id in emotion_map:
                y.append(emotion_map[emotion_id])

        except Exception as e:
            print(f"Hata: {file_path} -> {e}")

X = np.array(X)

# === Etiket encode ===
le = LabelEncoder()
y_enc = to_categorical(le.fit_transform(y))

np.save("classes.npy", le.classes_)

print(f"✅ Sınıflar: {list(le.classes_)}")

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y)

# === Sınıflandırıcı ===
model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(y_train.shape[1], activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer=Adam(0.0001), metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

# === Kaydet ===
model.save("wav2vec2_model.h5")
print("✅ wav2vec2_model.h5 ve classes.npy kaydedildi!")
