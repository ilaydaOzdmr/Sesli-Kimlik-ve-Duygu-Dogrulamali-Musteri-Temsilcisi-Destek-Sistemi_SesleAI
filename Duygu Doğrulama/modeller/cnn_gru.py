# cnn_gru_final.py

import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, GRU, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# 1. Parametreler
# -----------------------------
DATA_PATH =r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\Audio_Speech_Actors_01-24 (1)"
n_mfcc = 40
dropout_rate = 0.3

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
# 7. CNN + GRU Modeli
# -----------------------------
model = Sequential()
# Conv1D + BatchNorm + Pooling + Dropout
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(dropout_rate))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(dropout_rate))

# GRU katmanları
model.add(GRU(128, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(GRU(64))
model.add(Dropout(dropout_rate))

# Output
model.add(Dense(y_train.shape[1], activation='softmax'))

# -----------------------------
# 8. Compile
# -----------------------------
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# -----------------------------
# 9. Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('cnn_gru_best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# -----------------------------
# 10. Model Eğitimi
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)

# -----------------------------
# 11. Final modeli kaydet
# -----------------------------
model.save("cnn_gru_final_model.h5")
print("CNN+GRU modeli başarıyla eğitildi ve kaydedildi!")
