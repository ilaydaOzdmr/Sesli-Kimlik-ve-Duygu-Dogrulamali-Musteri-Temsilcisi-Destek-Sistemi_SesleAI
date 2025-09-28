import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, TimeDistributed, Flatten, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

#STATISTICSPOOLING KATMANI
class StatisticsPooling(Layer):
    def __init__(self, **kwargs):
        super(StatisticsPooling, self).__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=1)
        mean = tf.squeeze(mean, axis=1)
        return tf.concat([mean, variance], axis=1)

    def get_config(self):
        return super(StatisticsPooling, self).get_config()


# Veri Artırımı
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return (y + noise_factor * noise).astype(y.dtype)

def shift_time(y, shift_max=0.2):
    shift = np.random.randint(int(len(y) * shift_max))
    return np.roll(y, shift)

def change_speed(y, sr, speed_factor_range=(0.9, 1.1)):
    speed_factor = np.random.uniform(*speed_factor_range)
    try:
        return librosa.effects.time_stretch(y, rate=speed_factor)
    except Exception:
        return y

#  Özellik Çıkarma
def extract_features(file_path, n_mfcc=40, max_pad_len=100):
    if not os.path.exists(file_path):
        return None
    try:
        y, sr = librosa.load(file_path, sr=None)
        if np.random.rand() < 0.3:
            y = add_noise(y)
        if np.random.rand() < 0.3:
            y = shift_time(y)
        if np.random.rand() < 0.3:
            y = change_speed(y, sr)
        
        def pad_or_trim_feature(feature):
            if feature.shape[1] < max_pad_len:
                pad_width = max_pad_len - feature.shape[1]
                return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                return feature[:, :max_pad_len]

        mfccs = pad_or_trim_feature(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc))
        chroma = pad_or_trim_feature(librosa.feature.chroma_stft(y=y, sr=sr))
        contrast = pad_or_trim_feature(librosa.feature.spectral_contrast(y=y, sr=sr))
        zcr = pad_or_trim_feature(librosa.feature.zero_crossing_rate(y))

        features = np.vstack([mfccs, chroma, contrast, zcr])
        return features.T
    except Exception:
        return None

df_validated = pd.read_csv('validated.tsv', sep='\t')
min_clips_per_speaker = 5
speaker_counts = df_validated['client_id'].value_counts()
eligible_speakers = speaker_counts[speaker_counts >= min_clips_per_speaker].index
df_filtered = df_validated[df_validated['client_id'].isin(eligible_speakers)].copy()

label_encoder = LabelEncoder()
df_filtered['speaker_label'] = label_encoder.fit_transform(df_filtered['client_id'])
num_speakers = len(label_encoder.classes_)
n_mfcc = 40
max_pad_len = 100
total_features = n_mfcc + 12 + 7 + 1
input_shape = (max_pad_len, total_features)

X, y, skipped_count = [], [], 0
for _, row in df_filtered.iterrows():
    audio_file_path = os.path.join('clips', row['path'])
    features = extract_features(audio_file_path, n_mfcc, max_pad_len)
    if features is not None:
        X.append(features)
        y.append(row['speaker_label'])
    else:
        skipped_count += 1

X, y = np.array(X, dtype='float32'), np.array(y, dtype='int32')
if len(X) == 0:
    sys.exit("Hata: Hiçbir ses klibi işlenemedi.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model
x_vector_model = Sequential([
    tf.keras.layers.Input(shape=input_shape),
    TimeDistributed(Dense(512)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(LeakyReLU(alpha=0.1)),
    TimeDistributed(Dense(512)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(LeakyReLU(alpha=0.1)),
    TimeDistributed(Dense(512)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(LeakyReLU(alpha=0.1)),
    TimeDistributed(Dense(512)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(LeakyReLU(alpha=0.1)),
    StatisticsPooling(),
    Dense(512),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(512),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(num_speakers, activation='softmax')
])

x_vector_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

#Eğitme
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = x_vector_model.fit(X_train, y_train,
                             epochs=50,
                             batch_size=32,
                             validation_split=0.2,
                             callbacks=[early_stopping],
                             verbose=1)

model_path_h5 = 'sesli_kimlik_x_vector_model.h5'
x_vector_model.save(model_path_h5)

print("\n--- Model Test Sonuçları ---")
loss, accuracy = x_vector_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")
print(f"Son Eğitim Doğruluğu: {history.history['accuracy'][-1]:.4f}")
print(f"Son Doğrulama Doğruluğu: {history.history['val_accuracy'][-1]:.4f}")

print("\n--- Sınıflandırma Raporu ---")
y_pred_probs = x_vector_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix (X-vector Modeli)')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Doğruluk Eğrisi (X-vector Modeli)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp Eğrisi (X-vector Modeli)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.tight_layout()
plt.show()
