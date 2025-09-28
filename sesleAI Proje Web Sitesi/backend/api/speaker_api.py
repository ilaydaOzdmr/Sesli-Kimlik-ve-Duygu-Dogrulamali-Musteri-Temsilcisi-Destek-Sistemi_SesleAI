import os
import torch
import torch.nn as nn
import torchaudio
import json
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import io

# Bu dosyanın bulunduğu klasörün mutlak yolunu al
API_DIR = os.path.dirname(os.path.abspath(__file__))

# GPU varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeakerRecognitionAPI:
    def __init__(self):
        self.models_path = os.path.join(API_DIR, '../models')
        self.db_path = os.path.join(API_DIR, 'speakers_db.json')
        
        # Sadece encoder kısmını yüklüyoruz. Sınıflandırıcı artık kullanılmayacak.
        self.encoder_classifier = self._load_encoder()
        self.speakers_db = {}
        self._load_speakers_db()
        self._update_label_encoder()

    def _load_encoder(self):
        print("Model yükleniyor...")
        try:
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir=os.path.join(self.models_path, "pretrained-ecapa"),
                run_opts={"device": device}
            )
            model.to(device)
            model.eval()
            print("Model başarıyla yüklendi.")
            return model
        except Exception as e:
            print(f"Hata: Model yüklenirken bir sorun oluştu. Detay: {e}")
            raise

    def _load_speakers_db(self):
        if os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0:
            with open(self.db_path, 'r') as f:
                self.speakers_db = json.load(f)
        else:
            self.speakers_db = {}
        
    def _update_label_encoder(self):
        if not self.speakers_db:
            self.label_encoder = None
            return
        
        speakers = list(self.speakers_db.keys())
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(speakers)

    def _extract_embedding(self, audio_path):
        try:
            # Ses dosyasını soundfile ile oku (daha esnek)
            audio, sr = sf.read(audio_path)
            # numpy array'ini torch tensor'a dönüştür
            signal = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                embeddings = self.encoder_classifier.encode_batch(signal).squeeze()
            return embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"Embedding çıkarma hatası: {e}")
            raise

    def register_speaker_with_multiple_files(self, name, audio_paths):
        if name in self.speakers_db:
            return False, "Bu isimde bir konuşmacı zaten kayıtlı."

        embeddings = []
        for path in audio_paths:
            embedding = self._extract_embedding(path)
            embeddings.append(embedding.tolist())
        
        self.speakers_db[name] = embeddings
        
        with open(self.db_path, 'w') as f:
            json.dump(self.speakers_db, f, indent=4)

        self._update_label_encoder()
        return True, "Konuşmacı başarıyla kaydedildi."
        
    def update_speaker(self, name, audio_paths):
        if name not in self.speakers_db:
            return False, "Bu isimde bir konuşmacı kayıtlı değil."
        
        for path in audio_paths:
            new_embedding = self._extract_embedding(path)
            self.speakers_db[name].append(new_embedding.tolist())
        
        with open(self.db_path, 'w') as f:
            json.dump(self.speakers_db, f, indent=4)

        self._update_label_encoder()
        return True, "Konuşmacı başarıyla güncellendi."

    def recognize_speaker(self, audio_path, threshold=0.85):
        if not self.speakers_db:
            return False, {"error": "Sistemde kayıtlı konuşmacı bulunamadı. Lütfen önce birini kaydedin."}
        
        new_embedding = self._extract_embedding(audio_path)
        
        max_similarity = -1.0
        predicted_speaker = "Bilinmiyor"
        
        for speaker_name, embeddings_list in self.speakers_db.items():
            for embedding_from_db in embeddings_list:
                db_embedding_np = np.array(embedding_from_db)
                similarity = np.dot(new_embedding, db_embedding_np) / (np.linalg.norm(new_embedding) * np.linalg.norm(db_embedding_np))
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_speaker = speaker_name
        
        if max_similarity > threshold:
            return True, {"speaker": predicted_speaker, "confidence": max_similarity}
        else:
            return False, {"message": "Konuşmacı tanınamıyor. Veritabanına kayıtlı değil.", "confidence": max_similarity}
            
    def correct_guess(self, name, audio_paths):
        if name not in self.speakers_db:
            return False, f"Doğru olduğunu belirttiğiniz '{name}' isimli konuşmacı kayıtlı değil."
        
        return self.update_speaker(name, audio_paths)