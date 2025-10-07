import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from speaker_api import SpeakerRecognitionAPI

# FastAPI uygulamasını başlat
app = FastAPI()

# CORS ayarları
origins = [
    "http://localhost:5173",  # Front end adresini buraya ekleyin
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    api = SpeakerRecognitionAPI()
except Exception as e:
    print(f"API başlatılırken bir hata oluştu: {e}")
    api = None

@app.get("/")
def read_root():
    return {"message": "Ses Kimlik Tespiti API'si çalışıyor."}

@app.post("/register/")
async def register_speaker(name: str, audio_files: list[UploadFile] = File(...)):
    if not api:
        return JSONResponse(status_code=500, content={"error": "API başlatılamadı."})

    temp_audio_paths = []
    for audio_file in audio_files:
        temp_audio_path = os.path.join(os.getcwd(), audio_file.filename)
        with open(temp_audio_path, "wb") as f:
            f.write(await audio_file.read())
        temp_audio_paths.append(temp_audio_path)

    success, message = api.register_speaker_with_multiple_files(name, temp_audio_paths)

    for temp_path in temp_audio_paths:
        os.remove(temp_path)

    if success:
        return JSONResponse(status_code=200, content={"message": message})
    else:
        return JSONResponse(status_code=400, content={"error": message})

@app.post("/recognize/")
async def recognize_speaker(audio_file: UploadFile = File(...)):
    if not api:
        return JSONResponse(status_code=500, content={"error": "API başlatılamadı."})
    
    temp_audio_path = os.path.join(os.getcwd(), audio_file.filename)
    with open(temp_audio_path, "wb") as f:
        f.write(await audio_file.read())

    success, result = api.recognize_speaker(temp_audio_path)

    os.remove(temp_audio_path)

    if success:
        return JSONResponse(status_code=200, content=result)
    else:
        return JSONResponse(status_code=404, content={"error": result})

@app.post("/update_speaker/")
async def update_speaker(name: str, audio_files: list[UploadFile] = File(...)):
    if not api:
        return JSONResponse(status_code=500, content={"error": "API başlatılamadı."})
    
    temp_audio_paths = []
    for audio_file in audio_files:
        temp_audio_path = os.path.join(os.getcwd(), audio_file.filename)
        with open(temp_audio_path, "wb") as f:
            f.write(await audio_file.read())
        temp_audio_paths.append(temp_audio_path)

    success, message = api.update_speaker(name, temp_audio_paths)

    for temp_path in temp_audio_paths:
        os.remove(temp_path)

    if success:
        return JSONResponse(status_code=200, content={"message": message})
    else:
        return JSONResponse(status_code=400, content={"error": message})

@app.post("/correct_guess/")
async def correct_guess(name: str, audio_file: UploadFile = File(...)):
    if not api:
        return JSONResponse(status_code=500, content={"error": "API başlatılamadı."})
    
    temp_audio_path = os.path.join(os.getcwd(), audio_file.filename)
    with open(temp_audio_path, "wb") as f:
        f.write(await audio_file.read())

    success, message = api.correct_guess(name, [temp_audio_path])

    os.remove(temp_audio_path)

    if success:
        return JSONResponse(status_code=200, content={"message": message})
    else:
        return JSONResponse(status_code=400, content={"error": message})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
