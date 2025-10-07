import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle

# -----------------------------
# 1. Parametreler
# -----------------------------
DATASET_PATH = "C:/Users/kdrt2/OneDrive/Masaüstü/model2/Audio_Speech_Actors_01-24 (1)/"
n_mfcc = 40
max_len = 100  # sequence length için padding

# -----------------------------
# 2. Veri Yükleme ve MFCC
# -----------------------------
X, y = [], []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        if file.endswith(".wav"):
            file_path = os.path.join(label_path, file)
            signal, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T
            # padding / truncate
            if mfcc.shape[0] > max_len:
                mfcc = mfcc[:max_len, :]
            else:
                pad_width = max_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            X.append(mfcc)
            y.append(label)

X = np.array(X)
y = np.array(y)

# -----------------------------
# 3. Label Encoding
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# -----------------------------
# 4. Normalizasyon
# -----------------------------
scaler = StandardScaler()
nsamples, nx, ny = X.shape
X = X.reshape((nsamples * nx, ny))
X = scaler.fit_transform(X)
X = X.reshape((nsamples, nx, ny))

# -----------------------------
# 5. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# -----------------------------
# 6. RNN Modeli
# -----------------------------
model = Sequential()
model.add(SimpleRNN(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 7. Callbacks
# -----------------------------
checkpoint = ModelCheckpoint("rnn_model_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -----------------------------
# 8. Model Eğitimi
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# 9. Test
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Test doğruluğu:", round(acc, 3))

# -----------------------------
# 10. Son olarak modeli kaydet
# -----------------------------
model.save("rnn_model_final.h5")
print("RNN modeli başarıyla kaydedildi!")
