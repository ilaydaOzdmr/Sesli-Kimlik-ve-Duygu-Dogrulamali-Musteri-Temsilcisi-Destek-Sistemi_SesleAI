import os
import glob
import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torchaudio
import numpy as np
import json

# GPU varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_and_train_model():
    """
    LibriSpeech verisiyle modeli eğitir ve daha sonra API üzerinden gelecek verilere hazır hale getirir.
    """
    models_path = os.path.join(os.getcwd(), 'models')
    db_path = os.path.join(os.getcwd(), 'api', 'speakers_db.json')
    classifier_path = os.path.join(models_path, 'speaker_classifier.pth')
    data_path = os.path.join(os.getcwd(), 'data')
    
    print("Veri setiniz taranıyor ve hazırlanıyor...")
    raw_data = {}
    subsets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    relevant_subsets = [s for s in subsets if 'train' in s or 'dev' in s]
    
    for subset in relevant_subsets:
        subset_path = os.path.join(data_path, subset)
        audio_files = glob.glob(os.path.join(subset_path, '**/*.flac'), recursive=True)
        for file_path in audio_files:
            speaker_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            utt_id = os.path.relpath(file_path, data_path)
            raw_data[utt_id] = {'wav': file_path, 'spk_id': speaker_id}

    if not raw_data:
        print("Eğitim için yeterli veri bulunamadı.")
        return

    speaker_ids = [item['spk_id'] for item in raw_data.values()]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(speaker_ids)
    unique_speakers = label_encoder.classes_
    num_classes = len(unique_speakers)
    
    print(f"Toplam {num_classes} eşsiz konuşmacı için model eğitilecek.")
    
    # Model yükleniyor
    print("Önceden eğitilmiş ECAPA-TDNN modeli yükleniyor...")
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=os.path.join(models_path, "pretrained-ecapa"),
            run_opts={"device": device}
        )
        model.to(device)
        model.eval()
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: Model yüklenirken bir sorun oluştu. Detay: {e}")
        return

    embeddings_dict = {}
    labels_list = []
    
    print("Embeddingler çıkarılıyor...")
    
    for uttid, data_point in raw_data.items():
        speaker_id = data_point['spk_id']
        audio_file_path = data_point['wav']
        
        try:
            signal, fs = torchaudio.load(os.path.join(data_path, audio_file_path))
            signal = signal.to(device)
            with torch.no_grad():
                embedding = model.encode_batch(signal).squeeze()
            
            if speaker_id not in embeddings_dict:
                embeddings_dict[speaker_id] = []
            embeddings_dict[speaker_id].append(embedding.detach().cpu().numpy())
            
            labels_list.append(label_encoder.transform([speaker_id])[0])
        except Exception as e:
            print(f"Hata: {audio_file_path} dosyası işlenirken bir sorun oluştu. Hata: {e}")

    # Her konuşmacı için ortalama embedding'i hesapla
    final_db = {}
    for speaker, embedding_list in embeddings_dict.items():
        avg_embedding = np.mean(embedding_list, axis=0).tolist()
        final_db[speaker] = avg_embedding
    
    # Veritabanını kaydet
    with open(db_path, 'w') as f:
        json.dump(final_db, f, indent=4)
    print(f"Veritabanı başarıyla dolduruldu. {len(final_db)} konuşmacı kaydedildi.")

    # Sınıflandırıcıyı Eğitme
    print("Sınıflandırıcı eğitiliyor...")
    embeddings = np.array(list(final_db.values()))
    labels = np.array(label_encoder.transform(list(final_db.keys())))
    
    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    classifier = nn.Linear(512, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = classifier(embedding_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Sınıflandırıcı Epoch [{epoch+1}/100], Kayıp: {loss.item():.4f}")

    torch.save(classifier.state_dict(), classifier_path)
    print("Sınıflandırıcı başarıyla eğitildi ve kaydedildi.")

if __name__ == "__main__":
    prepare_and_train_model()
