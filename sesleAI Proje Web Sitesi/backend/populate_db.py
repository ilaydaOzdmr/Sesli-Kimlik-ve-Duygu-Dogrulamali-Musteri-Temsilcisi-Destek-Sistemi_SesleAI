import os
import glob
import torch
import torch.nn as nn
import torchaudio
import json
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Bu dosyanın bulunduğu klasörün mutlak yolunu al
API_DIR = os.path.dirname(os.path.abspath(__file__))

# GPU varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(data_path):
    """
    LibriSpeech veri setini tarar ve eğitim için gerekli sözlük formatına dönüştürür.
    """
    print("Veri setiniz taranıyor ve hazırlanıyor...")
    data_dict = {}
    
    subsets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    relevant_subsets = [s for s in subsets if 'train' in s or 'dev' in s]
    
    for subset in relevant_subsets:
        subset_path = os.path.join(data_path, subset)
        audio_files = glob.glob(os.path.join(subset_path, '**/*.flac'), recursive=True)
        
        for file_path in audio_files:
            speaker_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            utt_id = os.path.relpath(file_path, data_path)
            
            data_dict[utt_id] = {
                'wav': file_path,
                'spk_id': speaker_id
            }

    print(f"Toplam {len(data_dict)} ses dosyası bulundu.")
    return data_dict

def extract_and_save_embeddings():
    """
    Eğitim veri setinden embeddingleri çıkarır, sınıflandırıcıyı eğitir ve kaydeder.
    """
    models_path = os.path.join(os.getcwd(), 'models')
    db_path = os.path.join(os.getcwd(), 'api', 'speakers_db.json')
    classifier_path = os.path.join(models_path, 'speaker_classifier.pth')
    data_path = os.path.join(os.getcwd(), 'data')
    
    raw_data = prepare_data(data_path)

    if not raw_data:
        print("Eğitim için yeterli veri bulunamadı. Veritabanı doldurulamadı.")
        return

    speaker_ids = [item['spk_id'] for item in raw_data.values()]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(speaker_ids)
    
    unique_speakers = label_encoder.classes_
    num_classes = len(unique_speakers)
    
    print(f"Toplam {num_classes} eşsiz konuşmacı kaydedilecek ve sınıflandırıcı eğitilecek.")
    
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=os.path.join(models_path, "pretrained-ecapa"),
            run_opts={"device": device}
        )
        encoder_state_dict = torch.load(os.path.join(models_path, 'speaker_encoder.pth'), map_location=device)
        model.load_state_dict(encoder_state_dict, strict=False)
        model.to(device)
        model.eval()
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: Model yüklenirken bir sorun oluştu. Detay: {e}")
        return

    embeddings = []
    labels = []
    speakers_db = {}
    print("Embeddingler çıkarılıyor...")
    
    for uttid, data_point in raw_data.items():
        speaker_id = data_point['spk_id']
        audio_file_path = data_point['wav']
        
        try:
            signal, fs = torchaudio.load(os.path.join(data_path, audio_file_path))
            signal = signal.to(device)
            with torch.no_grad():
                embedding = model.encode_batch(signal).squeeze()
            
            # Her embeddingi ilgili konuşmacının altına kaydet
            if speaker_id not in speakers_db:
                speakers_db[speaker_id] = []
            speakers_db[speaker_id].append(embedding.detach().cpu().numpy().tolist())
            
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label_encoder.transform([speaker_id])[0])
        except Exception as e:
            print(f"Hata: {audio_file_path} dosyası işlenirken bir sorun oluştu. Hata: {e}")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Sınıflandırıcıyı Eğitme
    print("Sınıflandırıcı eğitiliyor...")
    
    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    classifier = nn.Linear(512, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(100):  # 100 epoch eğitim
        optimizer.zero_grad()
        outputs = classifier(embedding_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Sınıflandırıcı Epoch [{epoch+1}/100], Kayıp: {loss.item():.4f}")

    # Eğitilmiş sınıflandırıcıyı kaydetme
    torch.save(classifier.state_dict(), classifier_path)
    print("Sınıflandırıcı başarıyla kaydedildi.")

    # Veritabanını kaydetme
    final_db = {}
    for speaker, embedding_list in speakers_db.items():
        final_db[speaker] = embedding_list
        
    with open(db_path, 'w') as f:
        json.dump(final_db, f, indent=4)
        
    print(f"Veritabanı başarıyla dolduruldu. {len(final_db)} konuşmacı kaydedildi.")

if __name__ == "__main__":
    extract_and_save_embeddings()