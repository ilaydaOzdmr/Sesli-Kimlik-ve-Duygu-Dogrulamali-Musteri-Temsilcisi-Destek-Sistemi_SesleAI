import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed, dump, load
import librosa
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random
from multiprocessing import freeze_support


def extract_mfcc_features(file_path, n_mfcc=40, flatten=False):
    if not os.path.exists(file_path):
        return None
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if flatten:
            return mfccs.T.flatten()
        return mfccs.T
    except Exception:
        return None

def main():
    try:
        df_validated = pd.read_csv('validated.tsv', sep='\t')
    except FileNotFoundError:
        sys.exit("Hata: validated.tsv dosyası bulunamadı.")

    min_clips_per_speaker = 10
    speaker_counts = df_validated['client_id'].value_counts()
    eligible_speakers = speaker_counts[speaker_counts >= min_clips_per_speaker].index
    df_filtered = df_validated[df_validated['client_id'].isin(eligible_speakers)].copy()

    if df_filtered.empty:
        sys.exit(f"Uyarı: Hiçbir konuşmacı '{min_clips_per_speaker}' klipten fazla değil.")

    label_encoder = LabelEncoder()
    df_filtered['speaker_label'] = label_encoder.fit_transform(df_filtered['client_id'])
    num_speakers = len(label_encoder.classes_)
    n_mfcc = 40

    X_train, X_test, y_train, y_test = train_test_split(
        df_filtered['path'], df_filtered['speaker_label'], test_size=0.2, random_state=42, stratify=df_filtered['speaker_label']
    )

    # Veri İşleme
    n_jobs = 4
    train_features = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(extract_mfcc_features)(os.path.join('clips', p)) for p in X_train)
    test_features = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(extract_mfcc_features)(os.path.join('clips', p)) for p in X_test)
    train_features = [f for f in train_features if f is not None]
    test_features = [f for f in test_features if f is not None]
    y_train_filtered = y_train[np.array([f is not None for f in train_features])]
    y_test_filtered = y_test[np.array([f is not None for f in test_features])]

    # Model
    all_train_features_flat = np.concatenate(train_features)
    n_components = 128
    # reg_covar parametresini ekledim.
    ubm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=20, verbose=1, reg_covar=1e-6)
    ubm.fit(all_train_features_flat)
    dump(ubm, 'gmm_ubm_model.pkl')

    speaker_gmms = {}
    for speaker_label in np.unique(y_train_filtered):
        speaker_features = np.concatenate([f for f, l in zip(train_features, y_train_filtered) if l == speaker_label])
        speaker_gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=10, verbose=0, reg_covar=1e-6)
        speaker_gmm.weights_ = ubm.weights_
        speaker_gmm.means_ = ubm.means_
        speaker_gmm.covariances_ = ubm.covariances_
        speaker_gmm.fit(speaker_features)
        speaker_gmms[speaker_label] = speaker_gmm
    dump(speaker_gmms, 'speaker_gmms.pkl')


    #Değerlendirme ve Sonuçlar
    y_pred = []
    for i, test_feature in enumerate(test_features):
        scores = np.array([gmm.score(test_feature) for gmm in speaker_gmms.values()])
        pred_label = np.argmax(scores)
        y_pred.append(pred_label)

    y_pred = np.array(y_pred)

    print("\n--- Model Test Sonuçları ---")
    accuracy = accuracy_score(y_test_filtered, y_pred)
    print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")

    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test_filtered, y_pred, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_test_filtered, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix (MFCC + GMM-UBM Modeli)')
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()