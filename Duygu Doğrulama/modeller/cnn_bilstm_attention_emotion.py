import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout,
                                     BatchNormalization, Bidirectional, LSTM,
                                     GlobalAveragePooling1D, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight

# ====== Ayarlar ======
dataset_path = r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\Audio_Speech_Actors_01-24 (1)"
n_mfcc = 40

# ====== Duygu Sınıfları ======
CLASSES = ["neutral","calm","happy","sad","angry","fear","disgust","surprise"]

# ====== Veri Yükleme ve Augmentasyon ======
X, y = [], []
actors = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

def augment_signal(signal, sr):
    # Time stretch
    stretched = librosa.effects.time_stretch(signal, rate=np.random.uniform(0.9, 1.1))
    # Pitch shift
    pitched = librosa.effects.pitch_shift(signal, sr, n_steps=np.random.randint(-2, 2))
    # Noise injection
    noise = signal + 0.005 * np.random.randn(len(signal))
    return [signal, stretched, pitched, noise]

for actor in actors:
    actor_path = os.path.join(dataset_path, actor)
    files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
    for file in files:
        file_path = os.path.join(actor_path, file)
        emotion_code = file.split("-")[2]
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
        
        signal, sr = librosa.load(file_path, sr=16000)
        augmented_signals = augment_signal(signal, sr)

        for sig in augmented_signals:
            mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc).T
            mel = librosa.feature.melspectrogram(y=sig, sr=sr).T
            chroma = librosa.feature.chroma_stft(y=sig, sr=sr).T
            features = np.hstack([
                mfcc,
                mel[:mfcc.shape[0], :40],
                chroma[:mfcc.shape[0], :12]
            ])
            X.append(features)
            y.append(emotion)

# ====== Padding ======
max_len = max([f.shape[0] for f in X])
X_pad = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# ====== Etiketleme ======
le = LabelEncoder()
y_enc = to_categorical(le.fit_transform(y))

# ====== Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# ====== Class Weights ======
class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(CLASSES)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights_values))

# ====== CNN + BiLSTM + MultiHeadAttention Modeli ======
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)

# MultiHeadAttention
x = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
x = GlobalAveragePooling1D()(x)

x = Dropout(0.3)(x)
outputs = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs, outputs)

# ====== Derleme ======
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# ====== Eğitim ======
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    class_weight=class_weights
)

# ====== Kaydetme ======
model.save("cnn_bilstm_multihead_model.h5")
print("Model kaydedildi: cnn_bilstm_multihead_model.h5")
