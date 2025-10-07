from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import uvicorn
import torch
import torchaudio
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging
import uuid
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ==========================
# Logging
# ==========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# FastAPI App
# ==========================
app = FastAPI(
    title="🎤 Wav2Vec2 Emotion Recognition API",
    description="Ses tabanlı duygu tanıma API'si - Wav2Vec2 modeli ile eğitilmiş",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Global Variables
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {}
LABEL_ENCODER = None
WAV2VEC_PROCESSOR = None
WAV2VEC_MODEL = None

# Turkish-English emotion mapping
EN_TO_TR = {
    "neutral": "Nötr",
    "calm": "Sakin", 
    "happy": "Mutlu",
    "sad": "Üzgün",
    "angry": "Kızgın",
    "fearful": "Endişeli",
    "disgust": "Hoşnutsuz",
    "surprised": "Şaşkın"
}

# ==========================
# Model Loading
# ==========================
def load_wav2vec_models():
    global WAV2VEC_PROCESSOR, WAV2VEC_MODEL
    try:
        WAV2VEC_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        WAV2VEC_MODEL.eval()
        logger.info("✅ Wav2Vec2 models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Wav2Vec2 models: {e}")
        return False

def load_classifier_model():
    global MODELS
    model_path = r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\apibackend_w2v\wav2vec2_model.h5"
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return False
    try:
        model = load_model(model_path, compile=False)
        MODELS["Wav2Vec2"] = model
        logger.info("✅ Wav2Vec2 classifier loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load classifier: {e}")
        return False


def load_label_encoder():
    global LABEL_ENCODER
    encoder_path = r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\classes.npy"
    if os.path.exists(encoder_path):
        try:
            LABEL_ENCODER = np.load(encoder_path)
            logger.info(f"✅ Label encoder loaded: {LABEL_ENCODER}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not load label encoder: {e}")
    else:
        logger.info("ℹ️ No label encoder found, using default classes")
    return False

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Wav2Vec2 Emotion Recognition API")
    load_wav2vec_models()
    load_classifier_model()
    load_label_encoder()
    logger.info("✅ API ready for predictions")

# ==========================
# Helpers
# ==========================
def convert_to_wav(input_path: str) -> str:
    """Convert any input (e.g. webm/opus) to WAV 16kHz mono"""
    try:
        output_path = input_path + "_conv.wav"
        sound = AudioSegment.from_file(input_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(output_path, format="wav")
        return output_path
    except Exception as e:
        logger.error(f"❌ Conversion to wav failed: {e}")
        return None

def extract_wav2vec_features(file_path: str) -> np.ndarray:
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0)  # mono
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        inputs = WAV2VEC_PROCESSOR(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = WAV2VEC_MODEL(inputs.input_values)
            hidden_states = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return hidden_states.squeeze()
    except Exception as e:
        logger.error(f"❌ Wav2Vec2 feature extraction failed: {e}")
        return None

def preprocess_for_wav2vec(features: np.ndarray) -> np.ndarray:
    if features is None or len(features.shape) == 0:
        return None
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    return features

def predict_emotion_segmented(audio_path: str, segment_length: int = 5000):
    """
    Ses dosyasını belirtilen uzunluktaki segmentlere ayırır (varsayılan 5 saniye).
    Her segmenti geçici bir dosya olarak kaydeder.
    predict_emotion() fonksiyonunu kullanarak her segment için duygu tahmini yapar.
    Sonuçları hem sırayla listeler hem de majority voting yöntemiyle genel duyguyu döndürür.
    
    Args:
        audio_path (str): Ses dosyası yolu
        segment_length (int): Segment uzunluğu (milisaniye, varsayılan 5000ms = 5sn)
    
    Returns:
        tuple: (results, final_emotion)
            - results: Her segmentin duygusu listesi
            - final_emotion: Majority voting sonucu genel duygu
    """
    try:
        # Ses dosyasını yükle
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio)  # milisaniye
        
        # logger.info(f"🎵 Ses dosyası yüklendi: {total_duration}ms ({total_duration/1000:.1f}sn)")
        
        results = []
        emotion_counts = {}
        temp_files = []
        
        # Segmentlere ayır
        for start_time in range(0, total_duration, segment_length):
            end_time = min(start_time + segment_length, total_duration)
            segment = audio[start_time:end_time]
            
            # Geçici dosya oluştur
            temp_file = f"temp_segment_{start_time}_{uuid.uuid4().hex}.wav"
            temp_path = os.path.join(BASE_DIR, temp_file)
            temp_files.append(temp_path)
            
            # Segmenti kaydet
            segment.export(temp_path, format="wav")
            
            # Duygu tahmini yap - doğrudan mevcut mantığı kullan
            try:
                # Convert to wav if needed
                wav_path = convert_to_wav(temp_path)
                if not wav_path:
                    continue

                # Extract features
                features = extract_wav2vec_features(wav_path)
                if features is None:
                    continue

                x = preprocess_for_wav2vec(features)
                if x is None:
                    continue

                model = MODELS.get("Wav2Vec2")
                if model is None:
                    continue

                prediction = model.predict(x, verbose=0)
                pred_idx = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                if LABEL_ENCODER is not None and pred_idx < len(LABEL_ENCODER):
                    predicted_emotion = LABEL_ENCODER[pred_idx]
                    prediction_tr = EN_TO_TR.get(predicted_emotion, predicted_emotion)
                else:
                    continue
                    
                prediction_result = {
                    "prediction": predicted_emotion, 
                    "prediction_tr": prediction_tr,
                    "confidence": confidence
                }
            except Exception as e:
                logger.warning(f"Segment {start_time}-{end_time}ms için hata: {e}")
                continue
            
            if "error" in prediction_result:
                logger.warning(f"⚠️ Segment {start_time}-{end_time}ms için hata: {prediction_result['error']}")
                continue
            
            # Sonucu kaydet
            emotion_tr = prediction_result.get("prediction_tr", "Bilinmiyor")
            confidence = prediction_result.get("confidence", 0.0)
            
            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "emotion": emotion_tr,
                "confidence": confidence
            })
            
            # Majority voting için say
            if emotion_tr in emotion_counts:
                emotion_counts[emotion_tr] += 1
            else:
                emotion_counts[emotion_tr] = 1
            
            # Sonucu yazdır (sadece gerekirse)
            # start_sec = start_time / 1000
            # end_sec = end_time / 1000
            # print(f"{start_sec:.0f}-{end_sec:.0f} sn: {emotion_tr} (güven: {confidence:.2f})")
        
        # Majority voting ile final duygu
        if emotion_counts:
            final_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            final_emotion = "Bilinmiyor"
        
        # logger.info(f"📊 Segment analizi tamamlandı. Final duygu: {final_emotion}")
        # logger.info(f"📈 Duygu dağılımı: {emotion_counts}")
        
        return results, final_emotion
        
    except Exception as e:
        logger.error(f"❌ Segment analizi hatası: {e}")
        return [], "Hata"
    finally:
        # Geçici dosyaları temizle
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

# ==========================
# Endpoints
# ==========================
@app.get("/")
async def root():
    return {
        "message": "🎤 Wav2Vec2 Emotion Recognition API",
        "status": "running",
        "models_loaded": list(MODELS.keys())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file or not file.filename:
        return {"error": "Dosya yüklenmedi"}

    temp_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4().hex}_{file.filename}")
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return {"error": f"Dosya kaydedilemedi: {str(e)}"}

    try:
        # 🔹 Zorunlu dönüşüm
        wav_path = convert_to_wav(temp_path)
        if not wav_path:
            return {"error": "Ses dönüştürülemedi"}

        # Özellik çıkar
        features = extract_wav2vec_features(wav_path)
        if features is None:
            return {"error": "Özellik çıkarılamadı"}

        x = preprocess_for_wav2vec(features)
        if x is None:
            return {"error": "Özellik işlenemedi"}

        model = MODELS.get("Wav2Vec2")
        if model is None:
            return {"error": "Model yüklenmedi"}

        prediction = model.predict(x, verbose=0)
        pred_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if LABEL_ENCODER is not None and pred_idx < len(LABEL_ENCODER):
            predicted_emotion = LABEL_ENCODER[pred_idx]
        else:
            return {"error": "Label encoder not loaded properly"}

        # Get Turkish translation
        prediction_tr = EN_TO_TR.get(predicted_emotion, predicted_emotion)

        return {
            "prediction": predicted_emotion, 
            "prediction_tr": prediction_tr,
            "confidence": confidence
        }



    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "Tahmin sırasında hata oluştu"}
    finally:
        for path in [temp_path, temp_path + "_conv.wav"]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    uvicorn.run("main_w2v:app", host="127.0.0.1", port=8001, reload=True, log_level="info")

