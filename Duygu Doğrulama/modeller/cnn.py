import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# ====== RAVDESS veri yolu ======
DATA_PATH = "C:/Users/kdrt2/OneDrive/Masaüstü/model2/Audio_Speech_Actors_01-24 (1)/"

# ====== Duygu kodları ======
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

X = []
y = []

print("Veri dosyaları kontrol ediliyor...")

for root, dirs, files in os.walk(DATA_PATH):
    wav_files = [f for f in files if f.endswith(".wav")]
    if len(wav_files) > 0:
        print(f"{root} klasöründe {len(wav_files)} adet .wav dosyası bulundu. Örnek dosyalar: {wav_files[:5]}")
    for file in wav_files:
        file_path = os.path.join(root, file)
        
        # Duygu kodunu 5. parçadan al
        emotion_code = file.split("-")[4]
        emotion = emotion_map.get(emotion_code)
        if emotion is None:
            continue
        
        # Ses yükleme
        signal, sr = librosa.load(file_path, sr=16000)
        
        # MFCC çıkarımı
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        X.append(mfcc_scaled)
        y.append(emotion)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("X dizisi boş kaldı! Dosya yolunu ve dosya formatlarını kontrol et.")

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# One-hot encoding
y_train_cat = to_categorical(y_train, num_classes=8)
y_test_cat = to_categorical(y_test, num_classes=8)

# ====== CNN Modeli ======
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(40,1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ====== Model eğitimi ======
X_train_cnn = X_train[..., np.newaxis]  # Conv1D için reshape
X_test_cnn = X_test[..., np.newaxis]

history = model.fit(X_train_cnn, y_train_cat, epochs=50, batch_size=32,
                    validation_data=(X_test_cnn, y_test_cat))

# ====== Modeli kaydet ======
model.save("CNN_model.h5")
print("CNN modeli kaydedildi: CNN_model.h5")
