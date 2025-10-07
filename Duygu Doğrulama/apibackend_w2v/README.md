# Wav2Vec2 Emotion Recognition Backend

FastAPI tabanlı duygu analizi backend servisi - Wav2Vec2 modeli ile.

## Kurulum

```bash
cd apibackend_w2v
pip install -r requirements.txt
```

## Model Dosyaları

Aşağıdaki dosyaların `modeller/` klasöründe bulunması gerekir:
- `Bab2Vec.h5` - Eğitilmiş Wav2Vec2 sınıflandırıcı modeli
- `classes.npy` - Sınıf etiketleri (opsiyonel)

## Çalıştırma

```bash
python main_w2v.py
```

API http://localhost:8001 adresinde çalışacak.

## Endpoints

- `GET /` - API bilgileri
- `GET /health` - Sağlık kontrolü
- `POST /predict` - Duygu analizi

## Özellikler

- Wav2Vec2 özellik çıkarımı
- 16kHz ses işleme
- 7 sınıf duygu tanıma
- JSON hata yönetimi
- CORS desteği
