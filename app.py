from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Konfigürasyon
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
USERS_FILE = 'users.json'
SAMPLE_RATE = 16000
FEATURE_DIM = 128

# Klasörleri oluştur
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

class VoiceIdentitySystem:
    def __init__(self):
        self.users = self.load_users()
        self.scaler = StandardScaler()
        self.gmm_models = {}
        self.load_models()
    
    def load_users(self):
        """Kullanıcı veritabanını yükle"""
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        """Kullanıcı veritabanını kaydet"""
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)
    
    def load_models(self):
        """Eğitilmiş modelleri yükle"""
        for user_id in self.users:
            model_path = os.path.join(MODELS_FOLDER, f"{user_id}_model.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.gmm_models[user_id] = pickle.load(f)
                except:
                    logger.error(f"Model yüklenemedi: {user_id}")
    
    def extract_features(self, audio_path):
        """Ses dosyasından özellik çıkar"""
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            
            # MFCC özelliklerini çıkar
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Delta ve delta-delta özellikler
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Özellikleri birleştir
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            # Ortalama ve standart sapma
            features_mean = np.mean(features, axis=1)
            features_std = np.std(features, axis=1)
            
            # Son özellik vektörü
            feature_vector = np.concatenate([features_mean, features_std])
            
            # Boyutu normalize et
            if len(feature_vector) > FEATURE_DIM:
                feature_vector = feature_vector[:FEATURE_DIM]
            elif len(feature_vector) < FEATURE_DIM:
                feature_vector = np.pad(feature_vector, (0, FEATURE_DIM - len(feature_vector)))
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Özellik çıkarma hatası: {e}")
            return None
    
    def register_user(self, user_id, name, audio_paths):
        """Yeni kullanıcı kaydet"""
        try:
            features_list = []
            
            for audio_path in audio_paths:
                features = self.extract_features(audio_path)
                if features is not None:
                    features_list.append(features)
            
            if len(features_list) < 3:
                return False, "En az 3 ses örneği gerekli"
            
            # Özellikleri normalize et
            features_array = np.array(features_list)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # GMM modeli eğit
            gmm = GaussianMixture(n_components=3, random_state=42)
            gmm.fit(features_scaled)
            
            # Modeli kaydet
            model_path = os.path.join(MODELS_FOLDER, f"{user_id}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(gmm, f)
            
            # Kullanıcı bilgilerini kaydet
            self.users[user_id] = {
                'name': name,
                'registered_date': datetime.now().isoformat(),
                'samples_count': len(features_list),
                'model_path': model_path
            }
            
            self.gmm_models[user_id] = gmm
            self.save_users()
            
            return True, "Kullanıcı başarıyla kaydedildi"
            
        except Exception as e:
            logger.error(f"Kullanıcı kayıt hatası: {e}")
            return False, str(e)
    
    def identify_user(self, audio_path):
        """Ses dosyasından kullanıcı kimliğini belirle"""
        try:
            features = self.extract_features(audio_path)
            if features is None:
                return None, "Özellik çıkarılamadı"
            
            features_scaled = self.scaler.transform([features])
            
            best_match = None
            best_score = -float('inf')
            
            for user_id, gmm in self.gmm_models.items():
                try:
                    score = gmm.score(features_scaled)
                    if score > best_score:
                        best_score = score
                        best_match = user_id
                except:
                    continue
            
            if best_match and best_score > -10:  # Eşik değeri
                return best_match, self.users[best_match]['name']
            else:
                return None, "Tanınmayan kullanıcı"
                
        except Exception as e:
            logger.error(f"Kimlik belirleme hatası: {e}")
            return None, str(e)
    
    def update_user_model(self, user_id, audio_paths):
        """Kullanıcı modelini güncelle"""
        try:
            if user_id not in self.users:
                return False, "Kullanıcı bulunamadı"
            
            # Mevcut özellikleri yükle
            existing_features = []
            for audio_path in audio_paths:
                features = self.extract_features(audio_path)
                if features is not None:
                    existing_features.append(features)
            
            if len(existing_features) < 2:
                return False, "En az 2 yeni ses örneği gerekli"
            
            # Yeni özellikleri ekle
            features_array = np.array(existing_features)
            features_scaled = self.scaler.transform(features_array)
            
            # Modeli güncelle
            gmm = self.gmm_models[user_id]
            gmm.fit(features_scaled)
            
            # Güncellenmiş modeli kaydet
            model_path = os.path.join(MODELS_FOLDER, f"{user_id}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(gmm, f)
            
            # Kullanıcı bilgilerini güncelle
            self.users[user_id]['samples_count'] += len(existing_features)
            self.users[user_id]['last_updated'] = datetime.now().isoformat()
            self.save_users()
            
            return True, "Model başarıyla güncellendi"
            
        except Exception as e:
            logger.error(f"Model güncelleme hatası: {e}")
            return False, str(e)

# Global sistem instance'ı
voice_system = VoiceIdentitySystem()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Sistem sağlık kontrolü"""
    try:
        return jsonify({
            'status': 'healthy',
            'users_count': len(voice_system.users),
            'models_loaded': len(voice_system.gmm_models)
        })
    except Exception as e:
        logger.error(f"Health check hatası: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    """Yeni kullanıcı kaydı"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        audio_paths = data.get('audio_paths', [])
        
        if not user_id or not name or not audio_paths:
            return jsonify({'success': False, 'error': 'Eksik parametreler'}), 400
        
        success, message = voice_system.register_user(user_id, name, audio_paths)
        
        return jsonify({
            'success': success,
            'message': message,
            'user_id': user_id if success else None
        })
        
    except Exception as e:
        logger.error(f"Kayıt endpoint hatası: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/identify', methods=['POST'])
def identify_user():
    """Kullanıcı kimlik doğrulama"""
    try:
        data = request.get_json()
        audio_path = data.get('audio_path')
        
        if not audio_path:
            return jsonify({'success': False, 'error': 'Ses dosyası yolu gerekli'}), 400
        
        user_id, user_name = voice_system.identify_user(audio_path)
        
        if user_id:
            return jsonify({
                'success': True,
                'user_id': user_id,
                'user_name': user_name,
                'message': f'Hoş geldiniz, {user_name}!'
            })
        else:
            return jsonify({
                'success': False,
                'message': user_name
            })
        
    except Exception as e:
        logger.error(f"Kimlik doğrulama endpoint hatası: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update', methods=['POST'])
def update_user():
    """Kullanıcı modelini güncelle"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        audio_paths = data.get('audio_paths', [])
        
        if not user_id or not audio_paths:
            return jsonify({'success': False, 'error': 'Eksik parametreler'}), 400
        
        success, message = voice_system.update_user_model(user_id, audio_paths)
        
        return jsonify({
            'success': success,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Güncelleme endpoint hatası: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Kayıtlı kullanıcıları listele"""
    try:
        users_list = []
        for user_id, user_data in voice_system.users.items():
            users_list.append({
                'user_id': user_id,
                'name': user_data['name'],
                'registered_date': user_data['registered_date'],
                'samples_count': user_data['samples_count']
            })
        
        return jsonify({
            'success': True,
            'users': users_list
        })
        
    except Exception as e:
        logger.error(f"Kullanıcı listesi endpoint hatası: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Ses dosyası yükleme"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'Ses dosyası bulunamadı'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'}), 400
        
        # Dosya adını güvenli hale getir
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Dosya yükleme hatası: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

