import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from itertools import combinations
import sys
import multiprocessing

#MODEL İŞLEME UZUN SÜRDÜĞÜ İÇİN SONUÇ ALINAMADI !!!!

# --- Gelişmiş Özellik Çıkarma Fonksiyonu ---
def extract_features(file_path, n_mfcc=40, max_pad_len=100):
    if not os.path.exists(file_path):
        return None
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        
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
        return features[..., np.newaxis]
    except Exception:
        return None

# --- Veri Ön İşleme (Çoklu İşlem ile) ---
def process_single_path(path_item):
    """Multiprocessing havuzu için tek bir ses dosyasını işler."""
    path, n_mfcc, max_pad_len = path_item
    features = extract_features(path, n_mfcc, max_pad_len)
    return features

def preprocess_data_with_multiprocessing(all_paths, n_mfcc, max_pad_len):
    """Tüm ses dosyalarını paralel olarak ön işler."""
    print("Veriler ön işleniyor... Bu biraz zaman alabilir.")
    
    pool_size = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.map(process_single_path, [(p, n_mfcc, max_pad_len) for p in all_paths])
    
    print("Veri ön işleme tamamlandı.")
    return results

# --- Veri Yükleme ve Çift Oluşturma ---
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
input_shape = (total_features, max_pad_len, 1)

data = {
    label: [os.path.join('clips', path) for path in group['path']]
    for label, group in df_filtered.groupby('speaker_label')
}

same_speaker_pairs, diff_speaker_pairs = [], []
for label, paths in data.items():
    if len(paths) >= 2:
        same_speaker_pairs.extend(list(combinations(paths, 2)))
all_labels = list(data.keys())
for _ in range(len(same_speaker_pairs)):
    l1, l2 = random.sample(all_labels, 2)
    p1 = random.choice(data[l1])
    p2 = random.choice(data[l2])
    diff_speaker_pairs.append((p1, p2))

# Çiftleri birleştirip etiketleme
X_pairs = same_speaker_pairs + diff_speaker_pairs
y_pairs = [1] * len(same_speaker_pairs) + [0] * len(diff_speaker_pairs)

# Rastgele karıştırma
random_indices = list(range(len(X_pairs)))
random.shuffle(random_indices)
X_pairs = [X_pairs[i] for i in random_indices]
y_pairs = [y_pairs[i] for i in random_indices]

# Tüm benzersiz dosya yollarını toplama
all_paths = list(set([p[0] for p in X_pairs] + [p[1] for p in X_pairs]))
path_to_features = {}

# Multiprocessing ile tüm özellikleri önceden hesapla
preprocessed_features = preprocess_data_with_multiprocessing(all_paths, n_mfcc, max_pad_len)

# Özellikleri bir sözlükte saklama
for path, features in zip(all_paths, preprocessed_features):
    if features is not None:
        path_to_features[path] = features

# Önceden işlenmiş özelliklerle veri setini oluşturma
X_processed_pairs = []
y_processed_pairs = []
for (p1, p2), label in zip(X_pairs, y_pairs):
    if p1 in path_to_features and p2 in path_to_features:
        X_processed_pairs.append((path_to_features[p1], path_to_features[p2]))
        y_processed_pairs.append(label)

if not X_processed_pairs:
    sys.exit("Hata: Ön işleme sonrası geçerli veri çifti kalmadı.")

X1_processed = np.array([p[0] for p in X_processed_pairs])
X2_processed = np.array([p[1] for p in X_processed_pairs])
y_processed = np.array(y_processed_pairs)

# Veriyi eğitim ve test setlerine ayırma
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1_processed, X2_processed, y_processed, test_size=0.2, random_state=42
)

# --- TF.data.Dataset Oluşturma ---
train_dataset = tf.data.Dataset.from_tensor_slices(((X1_train, X2_train), y_train)) \
    .shuffle(1024) \
    .batch(32) \
    .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(((X1_test, X2_test), y_test)) \
    .batch(32) \
    .prefetch(tf.data.AUTOTUNE)

# --- Siamese Ağ Mimarisini Oluşturma ---
def get_embedding_network(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    return Model(input_layer, x)

embedding_network = get_embedding_network(input_shape)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
processed_a = embedding_network(input_a)
processed_b = embedding_network(input_b)

distance = Lambda(lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))([processed_a, processed_b])
siamese_model = Model([input_a, input_b], distance)

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    sq_pred = tf.square(y_pred)
    margin_sq = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * sq_pred + (1 - y_true) * margin_sq)

siamese_model.compile(loss=contrastive_loss, optimizer='adam')

# --- Modeli Eğitme ve Kaydetme ---
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = siamese_model.fit(train_dataset,
                            epochs=30,
                            validation_data=test_dataset,
                            callbacks=[early_stopping],
                            verbose=1)

embedding_network.save('sesli_kimlik_siamese_embedding_model.h5')

# --- Sonuçları Yazdırma ve Görselleştirme ---
test_distances = siamese_model.predict(test_dataset)
test_labels = np.concatenate([y.numpy() for x, y in test_dataset])
threshold = np.mean(test_distances)
y_pred_labels = (test_distances <= threshold).astype('int32').flatten()

print("\n--- Model Test Sonuçları ---")
accuracy = np.mean(y_pred_labels == test_labels)
print(f"Eşik Değeri (Threshold): {threshold:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")

print("\n--- Sınıflandırma Raporu ---")
print(classification_report(test_labels, y_pred_labels, target_names=['Farklı Konuşmacı', 'Aynı Konuşmacı']))

cm = confusion_matrix(test_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Farklı', 'Aynı'],
            yticklabels=['Farklı', 'Aynı'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix (Siamese Network)')
plt.show()