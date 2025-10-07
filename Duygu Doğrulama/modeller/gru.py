# gru_model.py

import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# 1. Parametreler
DATA_PATH ="C:/Users/kdrt2/OneDrive/Masaüstü/model2/Audio_Speech_Actors_01-24 (1)/" # veri klasör yolu
n_mfcc = 40

# 2. Veri ve etiketleri toplama
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

# 3. Padding (tüm MFCC’leri aynı uzunluğa getir)
max_len = max([mfcc.shape[0] for mfcc in X])
X_pad = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# 4. Etiketleri one-hot encode et
le = LabelEncoder()
y_enc = to_categorical(le.fit_transform(y))

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_enc, test_size=0.2, random_state=42)

# 6. GRU Modeli
model = Sequential()
model.add(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.4))
model.add(GRU(64))
model.add(Dropout(0.4))
model.add(Dense(y_train.shape[1], activation='softmax'))

# 7. Modeli derle
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 8. Eğit
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------
# Callback tanımları
# -------------------
early_stop = EarlyStopping(
    monitor='val_loss',  # izlenecek metrik
    patience=10,         # 10 epoch boyunca iyileşme olmazsa dur
    restore_best_weights=True  # en iyi ağırlıkları geri yükle
)

checkpoint = ModelCheckpoint(
    'gru_model_best.h5',   # kaydedilecek dosya
    monitor='val_accuracy',
    save_best_only=True
)

# -------------------
# Model Eğitimi
# -------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, checkpoint]  # buraya ekleniyor
)

# 9. Kaydet
model.save("gru_model.h5")

print("GRU modeli başarıyla eğitildi ve kaydedildi!")
