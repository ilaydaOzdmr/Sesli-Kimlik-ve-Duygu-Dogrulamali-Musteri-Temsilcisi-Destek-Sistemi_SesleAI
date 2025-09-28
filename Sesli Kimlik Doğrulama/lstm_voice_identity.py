import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import os
import matplotlib.pyplot as plt
import seaborn as sns




df_validated = pd.read_csv('validated.tsv', sep='\t')


# Her konuşmacı belirli bir minimum sayıda klip
min_clips_per_speaker = 5
speaker_counts = df_validated['client_id'].value_counts()
eligible_speakers = speaker_counts[speaker_counts >= min_clips_per_speaker].index
df_filtered = df_validated[df_validated['client_id'].isin(eligible_speakers)].copy()

if df_filtered.empty:
    print(f"Uyarı: Hiçbir konuşmacı '{min_clips_per_speaker}' klipten fazla değil. Lütfen 'min_clips_per_speaker' değerini düşürün veya veri setinizi kontrol edin.")
    exit()

# Konuşmacı kimliklerini sayısal etiketlere dönüştürme
label_encoder = LabelEncoder()
df_filtered['speaker_label'] = label_encoder.fit_transform(df_filtered['client_id'])
num_speakers = len(label_encoder.classes_)
print(f"Filtrelenmiş benzersiz konuşmacı sayısı: {num_speakers}")
print(f"Toplam kullanılacak klip sayısı: {len(df_filtered)}")

n_mfcc = 40        # MFCC katsayı
max_pad_len = 100  # Sabit zaman

# Veri Artırımı Fonksiyonları
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data.astype(y.dtype)

def shift_time(y, shift_max=0.2):
    shift = np.random.randint(int(len(y) * shift_max))
    return np.roll(y, shift)

def change_speed(y, sr, speed_factor_range=(0.9, 1.1)):
    speed_factor = np.random.uniform(*speed_factor_range)
    try:
        y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
        return y_stretched
    except Exception as e:
        return y # Hata durumunda orijinal sesi dön

# Gelişmiş Özellik Çıkarma(Data Augmentation vs)
def extract_features_with_augmentation(file_path, n_mfcc=40, max_pad_len=100):
    if not os.path.exists(file_path):
        return None

    try:
        y, sr = librosa.load(file_path, sr=None)

        # Rastgele augmentasyon
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
        
        features_stacked = np.vstack([mfccs, chroma, contrast, zcr])
        
        # LSTM için (max_pad_len, total_features) şekline transpoze edilmezse olmuyor
        return features_stacked.T

    except Exception as e:
        print(f"'{file_path}' dosyasında hata oluştu veya dosya bozuk: {e}. Bu klip atlanıyor.")
        return None
    
print(f"\nİşlenecek klip sayısı: {len(df_filtered)}.")

X = []
y = []
skipped_count = 0

for index, row in df_filtered.iterrows():
    audio_file_path = os.path.join('clips', row['path'])
    features = extract_features_with_augmentation(audio_file_path, n_mfcc, max_pad_len)

    if features is not None:
        X.append(features)
        y.append(row['speaker_label'])
    else:
        skipped_count += 1
    
    if (index + 1) % 100 == 0:
        print(f"{index + 1}/{len(df_filtered)} klip işlendi...")

print(f"\nToplam {len(X)} klip başarıyla işlendi ve yüklendi. {skipped_count} klip atlandı (dosya bulunamadı/hatalı).")

if len(X) == 0:
    print("Hata: Hiçbir ses klibi işlenemedi. Lütfen 'clips' klasörünüzün ve ses dosyalarının doğru olduğundan emin olun.")
    exit()

X = np.array(X, dtype='float32')
y = np.array(y, dtype='int32')

total_features = n_mfcc + 12 + 7 + 1 
input_shape_lstm = (max_pad_len, total_features) 

print(f"\nModelin giriş şekli: {input_shape_lstm}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
print(f"Test seti boyutu: {X_test.shape[0]} örnek")

# Bi-LSTM Modeli
model_lstm = Sequential([
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),

    Dense(num_speakers, activation='softmax')
])

model_lstm.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])


model_lstm.build(input_shape=(None, *input_shape_lstm)) # batch_sizeı None olarak bırakıyoruz

print("\n--- Oluşturulan Bi-LSTM Modelinin Özeti ---")
model_lstm.summary()


epochs = 30
batch_size = 32

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print(f"\nModel {epochs} epoch boyunca eğitiliyor (gerçek veri ile, EarlyStopping etkin)...")
history_lstm = model_lstm.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=0.2,
                              callbacks=[early_stopping],
                              verbose=1)

print("\nModel eğitimi tamamlandı.")

model_path_h5 = 'sesli_kimlik_bi_lstm_model.h5'
model_lstm.save(model_path_h5)


print("\n--- Model Test Sonuçları ---")
loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)
print(f"Test Kaybı (Loss): {loss_lstm:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy_lstm:.4f}")

print(f"Son Eğitim Doğruluğu: {history_lstm.history['accuracy'][-1]:.4f}")
print(f"Son Doğrulama Doğruluğu: {history_lstm.history['val_accuracy'][-1]:.4f}")

print("\n--- Sınıflandırma Raporu ---")
y_pred_probs_lstm = model_lstm.predict(X_test)
y_pred_lstm = np.argmax(y_pred_probs_lstm, axis=1)
print(classification_report(y_test, y_pred_lstm, target_names=label_encoder.classes_))

# Confusion Matrix
cm_lstm = confusion_matrix(y_test, y_pred_lstm)
plt.figure(figsize=(12,10))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix (Bi-LSTM Modeli)')
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_lstm.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Doğruluk Eğrisi (Bi-LSTM Modeli)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')

plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['loss'], label='Eğitim Kaybı')
plt.plot(history_lstm.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp Eğrisi (Bi-LSTM Modeli)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')

plt.tight_layout()
plt.show()
