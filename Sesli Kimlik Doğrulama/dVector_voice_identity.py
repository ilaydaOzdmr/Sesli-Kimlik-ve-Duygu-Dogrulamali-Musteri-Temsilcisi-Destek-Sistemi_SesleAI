import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
import multiprocessing

# Gelişmiş Özellik Çıkarma
def extract_features(file_path, n_mfcc=40, max_pad_len=100):
    if not os.path.exists(file_path):
        return None
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Basit augmentasyonlar (eğitim sırasında rastgele uygulanabilir)
        if np.random.rand() < 0.3:
            y += 0.005 * np.random.randn(len(y))
        if np.random.rand() < 0.3:
            y = np.roll(y, np.random.randint(int(len(y) * 0.2)))
        
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

# Veri Yükleme ve Hazırlık
try:
    df_validated = pd.read_csv('validated.tsv', sep='\t')
except FileNotFoundError:
    sys.exit("Hata: validated.tsv dosyası bulunamadı.")

min_clips_per_speaker = 5
speaker_counts = df_validated['client_id'].value_counts()
eligible_speakers = speaker_counts[speaker_counts >= min_clips_per_speaker].index
df_filtered = df_validated[df_validated['client_id'].isin(eligible_speakers)].copy()

if df_filtered.empty:
    sys.exit(f"Uyarı: Hiçbir konuşmacı '{min_clips_per_speaker}' klipten fazla değil.")

label_encoder = LabelEncoder()
df_filtered['speaker_label'] = label_encoder.fit_transform(df_filtered['client_id'])
num_speakers = len(label_encoder.classes_)
n_mfcc = 40
max_pad_len = 100
total_features = n_mfcc + 12 + 7 + 1 
input_shape = (max_pad_len, total_features)

df_train, df_test = train_test_split(df_filtered, test_size=0.2, random_state=42, stratify=df_filtered['speaker_label'])

def preprocess_data(path, label):
    def _py_extract_features(p):
        features = extract_features(p.numpy().decode('utf-8'), n_mfcc, max_pad_len)
        return features.astype('float32') if features is not None else np.zeros(input_shape, dtype='float32')
    
    features = tf.py_function(_py_extract_features, [path], tf.float32)
    features.set_shape(input_shape)
    
    return features, tf.cast(label, tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((
    [os.path.join('clips', p) for p in df_train['path']],
    df_train['speaker_label']
)).shuffle(1024).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    [os.path.join('clips', p) for p in df_test['path']],
    df_test['speaker_label']
)).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

# d-vector Modeli Oluşturma
def create_d_vector_model(input_shape, num_speakers):
    # Embedding network (d-vector extractor)
    embedding_input = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(embedding_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Bidirectional(LSTM(64))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    d_vector = Dense(128)(x) # 128 boyutlu d-vector (gömülü vektör)
    
    # Modelin embedding kısmını bir Model objesi olarak tanımlıyoruz
    embedding_model = Model(embedding_input, d_vector)
    
    # D-vector'dan sınıflandırma yapan kısmı ekliyoruz
    classifier_input = Input(shape=input_shape)
    embedding_output = embedding_model(classifier_input)
    
    x = Dense(256)(embedding_output)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    
    classification_output = Dense(num_speakers, activation='softmax')(x)
    
    # Ana sınıflandırma modelini oluşturuyoruz
    full_model = Model(classifier_input, classification_output)
    
    return embedding_model, full_model

embedding_model, full_model = create_d_vector_model(input_shape, num_speakers)

full_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Modeli Eğitme ve Kaydetme
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = full_model.fit(train_dataset,
                         epochs=30,
                         validation_data=test_dataset,
                         callbacks=[early_stopping],
                         verbose=1)

embedding_model.save('sesli_kimlik_d_vector_model.h5')

# Sonuçları Yazdırma ve Görselleştirme
print("\n--- Model Test Sonuçları ---")
loss, accuracy = full_model.evaluate(test_dataset, verbose=0)
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")

y_test_labels = np.concatenate([y.numpy() for x, y in test_dataset])
y_pred_probs = full_model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_test_labels, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix (d-vector Modeli)')
plt.show()