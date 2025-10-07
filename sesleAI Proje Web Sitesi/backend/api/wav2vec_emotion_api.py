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

# Mirror behavior of apibackend_w2v/main_w2v.py but inside this backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🎤 Wav2Vec2 Emotion Recognition API",
    description="Ses tabanlı duygu tanıma API'si - Wav2Vec2 modeli ile eğitilmiş",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {}
LABEL_ENCODER = None
WAV2VEC_PROCESSOR = None
WAV2VEC_MODEL = None

EN_TO_TR = {
    "neutral": "Nötr",
    "calm": "Sakin",
    "happy": "Mutlu",
    "sad": "Üzgün",
    "angry": "Kızgın",
    "fearful": "Endişeli",
    "disgust": "Hoşnutsuz",
    "surprised": "Şaşkın",
}

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
    # NOTE: Keep original external path reference if used by your environment.
    # You may want to move the .h5 next to this file and update path accordingly.
    model_path = os.environ.get(
        "W2V_CLASSIFIER_PATH",
        r"C:\\Users\\kdrt2\\OneDrive\\Masaüstü\\emotion-recognition-app\\apibackend_w2v\\wav2vec2_model.h5",
    )
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
    encoder_path = os.environ.get(
        "W2V_LABELS_PATH",
        r"C:\\Users\\kdrt2\\OneDrive\\Masaüstü\\emotion-recognition-app\\emotion-recognition-app1\\modeller\\classes.npy",
    )
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

def convert_to_wav(input_path: str) -> str:
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
        waveform = waveform.mean(dim=0)
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

@app.get("/")
async def root():
    return {
        "message": "🎤 Wav2Vec2 Emotion Recognition API",
        "status": "running",
        "models_loaded": list(MODELS.keys()),
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
        wav_path = convert_to_wav(temp_path)
        if not wav_path:
            return {"error": "Ses dönüştürülemedi"}

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

        prediction_tr = EN_TO_TR.get(predicted_emotion, predicted_emotion)
        return {"prediction": predicted_emotion, "prediction_tr": prediction_tr, "confidence": confidence}
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

if __name__ == "__main__":
    uvicorn.run("wav2vec_emotion_api:app", host="127.0.0.1", port=8001, reload=True, log_level="info")




