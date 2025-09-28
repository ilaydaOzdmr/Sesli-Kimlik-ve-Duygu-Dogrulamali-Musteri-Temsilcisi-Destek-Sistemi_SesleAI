import librosa
import numpy as np

def extract_features(file_path, num_mfcc=13, n_mels=40):
    """
    Ses dosyasından MFCC özniteliklerini çıkarır.

    Args:
        file_path (str): Ses dosyasının yolu.
        num_mfcc (int): Çıkarılacak MFCC sayısı.
        n_mels (int): Mel bankası sayısı.

    Returns:
        numpy.ndarray: Çıkarılan MFCC öznitelikleri.
                       Eğer dosya okuma hatası oluşursa None döner.
    """
    try:
        # Ses dosyasını librosa ile yükle
        audio, sr = librosa.load(file_path, sr=None)

        # Log-Mel spektrogramını oluştur
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)

        # MFCC'leri log-mel spektrogramından çıkar
        mfccs = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=num_mfcc)
        
        # Ortalama ve standart sapmasını alarak vektör haline getir
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Her iki vektörü birleştir
        features = np.concatenate((mfccs_mean, mfccs_std))

    except Exception as e:
        print(f"Hata: {file_path} dosyası işlenirken bir sorun oluştu. Hata: {e}")
        return None

    return features

if __name__ == '__main__':
    # Bu dosya tek başına çalıştırıldığında test etmek için
    sample_file_path = "ornek_ses_dosyasi.wav"
    
    # Varsayılan MFCC parametreleri ile öznitelikleri çıkar
    extracted_features = extract_features(sample_file_path)

    if extracted_features is not None:
        print("Öznitelikler başarıyla çıkarıldı.")
        print("Vektör boyutu:", extracted_features.shape)
        # print("Öznitelikler:", extracted_features)
    else:
        print("Öznitelik çıkarma işlemi başarısız oldu.")