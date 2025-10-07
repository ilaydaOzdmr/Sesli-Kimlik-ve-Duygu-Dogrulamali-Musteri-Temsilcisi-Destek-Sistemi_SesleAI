import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====== Ayarlar ======
dataset_path = r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\Audio_Speech_Actors_01-24 (1)"
n_mfcc = 40

# ====== Duygu Sınıfları (Backend ile uyumlu 8 sınıf) ======
CLASSES = ["neutral","calm","happy","sad","angry","fear","disgust","surprise"]

# ====== Veri Yükleme ======
X, y = [], []

actors = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
for actor in actors:
    actor_path = os.path.join(dataset_path, actor)
    files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
    for file in files:
        file_path = os.path.join(actor_path, file)
        
        # Dosya adından RAVDESS uyumlu duygu kodunu çıkar
        emotion_code = file.split("-")[2]  # Örn: '03' = happy
        emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fear",
            "07": "disgust",
            "08": "surprise"
        }
        emotion = emotion_map.get(emotion_code, "neutral")
        y.append(emotion)

        # MFCC çıkarımı
        signal, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T
        X.append(mfcc)

# ====== Padding ======
max_len = max([mfcc.shape[0] for mfcc in X])
X_pad = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# ====== Etiketleme ======
le = LabelEncoder()
y_enc = to_categorical(le.fit_transform(y))

# ====== Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_enc, test_size=0.2, random_state=42)

# ====== CNN + LSTM Modeli ======
model = Sequential()

# Conv1D katmanları
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

# LSTM katmanı
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(len(CLASSES), activation='softmax'))

# ====== Model Derleme ======
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# ====== Model Eğitimi ======
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# ====== Model Kaydetme ======
model.save("cnn_lstm_model.h5")
print("Model kaydedildi: cnn_lstm_model.h5")
