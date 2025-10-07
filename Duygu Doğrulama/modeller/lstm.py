import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")

# ====== Ayarlar ======
dataset_path = "C:/Users/kdrt2/OneDrive/Masaüstü/model2/Audio_Speech_Actors_01-24 (1)/"
n_mfcc = 40

# ====== Veri Yükleme ======
X, y = [], []

actors = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
for actor in actors:
    actor_path = os.path.join(dataset_path, actor)
    files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
    for file in files:
        file_path = os.path.join(actor_path, file)
        signal, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T
        X.append(mfcc)
        y.append(actor)

# ====== Padding ======
max_len = max([mfcc.shape[0] for mfcc in X])
X_pad = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# ====== Etiketleme ======
le = LabelEncoder()
y_enc = to_categorical(le.fit_transform(y))

# ====== Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_enc, test_size=0.2, random_state=42)

# ====== LSTM Modeli ======
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# ====== Model Eğitimi ======
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# ====== Model Kaydetme ======
model.save("lstm_model.h5")
print("Model kaydedildi: lstm_model.h5")
