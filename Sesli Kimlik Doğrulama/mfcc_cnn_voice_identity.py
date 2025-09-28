import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import os
import matplotlib.pyplot as plt
import seaborn as sns # Confusion Matrix çizimi için

 


df_validated = pd.read_csv('validated.tsv', sep='\t')


# Her konuşmacıdan belirli bir minimum sayıda klip olmasını sağla
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

# Özellik çıkarma ve doldurma/kesme için parametreler
n_mfcc = 40        # MFCC katsayı sayısı
max_pad_len = 100  # Sabit zaman boyutu

#Veri Artırımı
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data.astype(y.dtype) # Orijinal veri tipini koru

def shift_time(y, shift_max=0.2):
    shift = np.random.randint(int(len(y) * shift_max))
    return np.roll(y, shift)

def change_speed(y, sr, speed_factor_range=(0.9, 1.1)):
    speed_factor = np.random.uniform(*speed_factor_range)
    # Daha güvenli bir yaklaşım olarak y'nin uzunluğunu çarpanla bölerek yeni uzunluk hesaplıyoruz.
    if sr * (len(y) / sr / speed_factor) > 0:
        return librosa.effects.time_stretch(y, rate=speed_factor)
    return y # Geçersiz durumda orijinali dön

# Gelişmiş Özellik Çıkarma(Data Augmentation vs)
def extract_features_with_augmentation(file_path, n_mfcc=40, max_pad_len=100):
    if not os.path.exists(file_path):
        return None

    try:
        y, sr = librosa.load(file_path, sr=None)

        # Rastgele augmentasyon uygulama(En iyi gtest sonuçları %30 ile alındı.)
        if np.random.rand() < 0.3: # gürültü ekle
            y = add_noise(y)
        if np.random.rand() < 0.3: # zaman kaydır
            y = shift_time(y)
        if np.random.rand() < 0.3: # hızı değiştir
            y = change_speed(y, sr)
        
        # Özellik Çıkarımı
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Her özelliği max_pad_len zaman adımına göre pad et veya kes
        def pad_or_trim(feature):
            if feature.shape[1] < max_pad_len:
                pad_width = max_pad_len - feature.shape[1]
                return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                return feature[:, :max_pad_len]

        mfccs = pad_or_trim(mfccs)
        chroma = pad_or_trim(chroma)
        contrast = pad_or_trim(contrast)
        zcr = pad_or_trim(zcr)

        # ZCR genellikle (1, time_frames) şeklinde olur, bu yüzden onu da np.newaxis ile uyumlu hale getiririz.
        
        # zcr'yi (1, max_pad_len) şekline getir
        zcr_padded = pad_or_trim(zcr)

        features = np.vstack([mfccs, chroma, contrast, zcr_padded])
        
        return features[..., np.newaxis] # CNN için (toplam_features, max_pad_len, 1)

    except Exception as e:
        print(f"'{file_path}' dosyasında hata oluştu veya dosya bozuk: {e}. Bu klip atlanıyor.")
        return None

print(f"\nİşlenecek klip sayısı: {len(df_filtered)}.")

X = [] # Özellikler listesi
y = [] # Konuşmacı etiketleri listesi
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

# Toplam özellik sayısı (n_mfcc + chroma_bands + spectral_contrast_bands + zcr_bands)
# librosa.feature.chroma_stft varsayılan olarak 12 banda sahiptir.
# librosa.feature.spectral_contrast varsayılan olarak 7 banda sahiptir.
# zcr 1 banda sahiptir.
total_features = n_mfcc + 12 + 7 + 1 
input_shape = (total_features, max_pad_len, 1)

print(f"\nModelin giriş şekli: {input_shape}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
print(f"Test seti boyutu: {X_test.shape[0]} örnek")

# İyileştirilmiş CNN Modelini Oluşturma
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),

    Dense(num_speakers, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Oluşturulan İyileştirilmiş CNN Modelinin Özeti ---")
model.summary()


epochs = 30 # Optimim epoch 30 ile sağlandı 
batch_size = 32

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,           
    restore_best_weights=True # En iyi model ağırlıklarını geri yükle
)

print(f"\nModel {epochs} epoch boyunca eğitiliyor")
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)


print("\n--- Model Test Sonuçları ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")

print(f"Son Eğitim Doğruluğu: {history.history['accuracy'][-1]:.4f}")
print(f"Son Doğrulama Doğruluğu: {history.history['val_accuracy'][-1]:.4f}")

# Detaylı Sınıflandırma Raporu
print("\n--- Sınıflandırma Raporu ---")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
# label_encoder.classes_ kullanarak orijinal konuşmacı kimliklerini etiket olarak veriyoruz
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

model.save('mfcc_cnn_voice_identity_model.h5')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')

plt.tight_layout()
plt.show()

