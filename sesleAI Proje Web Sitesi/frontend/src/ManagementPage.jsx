import React, { useState, useRef } from 'react';
import './App.css';
import toWav from 'audiobuffer-to-wav';
import AudioVisualizer from './AudioVisualizer.jsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone, faStop, faUserPlus, faArrowLeft, faCheck, faTimes, faPlus, faX } from '@fortawesome/free-solid-svg-icons';
import logo from './DualMind_Logo.png';

const ManagementPage = ({ verification, onBack, onOpenEmotion }) => {
  // Seed subscribers: 5 Hat Sahibi + 5 Kullanıcı (same numbers, different names, same package)
  const SEED_SUBSCRIBERS = [
    { basePhone: '05327678989', role: 'owner', fullName: 'Mehmet Şimşek', packageName: 'GNC 30 GB' },
    { basePhone: '05327678989', role: 'user',  fullName: 'Derya Şimşek',  packageName: 'GNC 30 GB' },
    { basePhone: '05321234567', role: 'owner', fullName: 'Ayşe Yılmaz',    packageName: 'GNC 20 GB' },
    { basePhone: '05321234567', role: 'user',  fullName: 'Ali Yılmaz',     packageName: 'GNC 20 GB' },
    { basePhone: '05339876543', role: 'owner', fullName: 'Can Demir',      packageName: 'Platinum 40 GB' },
    { basePhone: '05339876543', role: 'user',  fullName: 'Elif Demir',     packageName: 'Platinum 40 GB' },
    { basePhone: '05325556677', role: 'owner', fullName: 'Fatma Kaya',     packageName: 'Super 10 GB' },
    { basePhone: '05325556677', role: 'user',  fullName: 'Mert Kaya',      packageName: 'Super 10 GB' },
    { basePhone: '05329991122', role: 'owner', fullName: 'Zeynep Acar',    packageName: 'GNC 8 GB' },
    { basePhone: '05418065249', role: 'owner', fullName: 'İlayda Özdemir',    packageName: 'GNC 30 GB' },
    { basePhone: '05329991122', role: 'user',  fullName: 'Burak Acar',     packageName: 'GNC 8 GB' },
  ].map(s => ({
    ...s,
    suffix: s.role === 'owner' ? 'HatSahibi' : 'Kullanici',
    combinedName: `${s.basePhone}${s.role === 'owner' ? 'HatSahibi' : 'Kullanici'}`,
  }));

  const [status, setStatus] = useState('Hazır');
  const [result, setResult] = useState(null);
  const [showPopup, setShowPopup] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const [audioData, setAudioData] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [formData, setFormData] = useState({
    fullName: '',
    phoneNumber: '',
    role: 'owner',
    packageName: 'GNC 30 GB'
  });
  const [subscribers, setSubscribers] = useState(() => {
    // localStorage'dan kayıtlı aboneleri yükle
    const saved = localStorage.getItem('subscribers');
    return saved ? JSON.parse(saved) : [];
  });
  const API_URL = 'http://localhost:8000';

  // Tüm aboneleri birleştir (SEED + Yeni eklenenler)
  const allSubscribers = [
    ...SEED_SUBSCRIBERS,
    ...subscribers
  ];
  const currentCombined = verification?.expectedName;
  const currentProfile = currentCombined
    ? allSubscribers.find(s => s.combinedName === currentCombined)
    : null;


  const handleRecordStart = async () => {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioData(stream);
      const chunks = [];
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/wav';
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });

      mediaRecorderRef.current.ondataavailable = (event) => { chunks.push(event.data); };
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: mediaRecorderRef.current.mimeType });
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioCtx = new AudioContext();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const wavBuffer = toWav(audioBuffer);
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        const wavFile = new File([wavBlob], 'recorded_audio.wav', { type: 'audio/wav' });
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        setAudioData(null);
        setSelectedFiles(prev => [...prev, wavFile]);
        setStatus('Kayıt eklendi. Göndermek için Kaydet butonunu kullanın.');
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setStatus('Kayıt başlatıldı...');
    } catch (err) {
      console.error('Mikrofon erişim hatası:', err);
      setStatus('Mikrofon erişim hatası!');
    }
  };

  const handleRecordStop = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setStatus('Kayıt durduruldu.');
    }
  };

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length > 0) {
      setSelectedFiles(prev => [...prev, ...files]);
      setStatus(`${files.length} dosya eklendi.`);
    }
  };

  const handleSave = async () => {
    if (!formData.fullName || !formData.phoneNumber || selectedFiles.length === 0) {
      setStatus('Lütfen tüm alanları doldurun ve ses kaydedin/yükleyin.');
      return;
    }
    const suffix = formData.role === 'owner' ? 'HatSahibi' : 'Kullanici';
    const combinedName = `${formData.phoneNumber}${suffix}`;
    const apiFormData = new FormData();
    apiFormData.append('name', combinedName);
    apiFormData.append('role', formData.role);
    selectedFiles.forEach((file, idx) => {
      const ext = (file.type && file.type.split('/')[1]) || 'wav';
      apiFormData.append('audio_files', file, `${formData.phoneNumber}_${idx}.${ext}`);
    });
    
     try {
       const response = await fetch(`${API_URL}/register/?name=${combinedName}&role=${formData.role}`, { method: 'POST', body: apiFormData });
       const data = await response.json();
       if (response.ok) {
         // Sözlük yapısına kaydet
         const newSubscriber = {
           basePhone: formData.phoneNumber,
           role: formData.role,
           fullName: formData.fullName,
           packageName: formData.packageName,
           suffix: suffix,
           combinedName: combinedName
         };
         
         setSubscribers(prev => {
           const updated = [...prev, newSubscriber];
           // localStorage'a kaydet
           localStorage.setItem('subscribers', JSON.stringify(updated));
           console.log('Yeni Eklenen Aboneler:', updated);
           console.log('Tüm Aboneler (SEED + Yeni):', [...SEED_SUBSCRIBERS, ...updated]);
           return updated;
         });
         
         setStatus(`Konuşmacı '${combinedName}' başarıyla kaydedildi!`);
         setResult({ type: 'success', message: `Konuşmacı '${combinedName}' başarıyla kaydedildi!` });
         setShowPopup(false);
         setSelectedFiles([]);
         setFormData({ fullName: '', phoneNumber: '', role: 'owner', packageName: 'GNC 30 GB' });
       } else {
         setStatus('Kayıt işlemi başarısız oldu.');
         setResult({ type: 'error', message: data.error || 'Bilinmeyen Hata' });
       }
     } catch (error) {
       console.error('Kayıt API çağrısı sırasında hata:', error);
       setStatus('Sunucuya bağlanırken hata oluştu.');
       setResult({ type: 'error', message: 'Sunucuya bağlanırken hata oluştu.' });
     }
  };


  return (
    <div className="App">
      <div className="container">
        <img src={logo} alt="Sesli AI Logosu" className="logo" />
        <h1>Konuşmacı Yönetimi(Müşteri Temsilcisi Sayfası)</h1>

        {currentProfile && (
          <div className="user-info">
            <span><strong>Ad Soyad:</strong> {currentProfile.fullName}</span>
            <span><strong>Telefon:</strong> {currentProfile.basePhone}</span>
            <span><strong>Paket:</strong> {currentProfile.packageName}</span>
            <span><strong>Rol:</strong> {currentProfile.role === 'owner' ? 'Hat Sahibi' : 'Kullanıcı'}</span>
          </div>
        )}

        {verification && (
          <div className={`result result-${verification.isVerified ? 'success' : 'error'}`}>
            {verification.isVerified ? (
              <>
                <FontAwesomeIcon icon={faCheck} style={{ marginRight: '8px' }} />
                {`Doğrulandı (%${verification.confidence})`}
              </>
            ) : (
              <>
                <FontAwesomeIcon icon={faTimes} style={{ marginRight: '8px' }} />
                {`Doğrulanamadı (%${verification.confidence})`}
              </>
            )}
          </div>
        )}

        <p className="subtitle">Yeni kayıt eklemek için aşağıdaki butona tıklayın.</p>

        <div className="controls">
          <button className="button" onClick={() => setShowPopup(true)}>
            <FontAwesomeIcon icon={faPlus} style={{ marginRight: '8px' }} />Kayıt Ekle
          </button>

          <button className="button" onClick={onOpenEmotion}>
            Duygu Doğrulama
          </button>

          <button className="button emotion-button" onClick={onBack}>
            <FontAwesomeIcon icon={faArrowLeft} style={{ marginRight: '8px' }} />Doğrulamaya Dön
          </button>
        </div>

        <div className="status">Durum: <span>{status}</span></div>
        {result && (<div className={`result result-${result.type}`}>{result.message}</div>)}

        {/* Popup Modal */}
        {showPopup && (
          <div className="popup-overlay">
            <div className="popup-content">
              <div className="popup-header">
                <h2>Yeni Kayıt Ekle</h2>
                <button className="close-button" onClick={() => setShowPopup(false)}>
                  <FontAwesomeIcon icon={faX} />
                </button>
              </div>
              
              <div className="popup-body">
                <div className="form-group">
                  <label>Ad Soyad:</label>
                  <input 
                    type="text" 
                    value={formData.fullName} 
                    onChange={(e) => setFormData({...formData, fullName: e.target.value})}
                    placeholder="Ad Soyad girin"
                  />
                </div>
                
                <div className="form-group">
                  <label>Telefon Numarası:</label>
                  <input 
                    type="text" 
                    value={formData.phoneNumber} 
                    onChange={(e) => setFormData({...formData, phoneNumber: e.target.value})}
                    placeholder="0532XXXXXXXX"
                  />
                </div>
                
                <div className="form-group">
                  <label>Rol:</label>
                  <select value={formData.role} onChange={(e) => setFormData({...formData, role: e.target.value})}>
                    <option value="owner">Hat Sahibi</option>
                    <option value="user">Kullanıcı</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label>Paket:</label>
                  <select value={formData.packageName} onChange={(e) => setFormData({...formData, packageName: e.target.value})}>
                    <option value="GNC 30 GB">GNC 30 GB</option>
                    <option value="GNC 20 GB">GNC 20 GB</option>
                    <option value="Platinum 40 GB">Platinum 40 GB</option>
                    <option value="Super 10 GB">Super 10 GB</option>
                    <option value="GNC 8 GB">GNC 8 GB</option>
                  </select>
                </div>

                <div className="audio-controls">
                  <div className="recording-controls">
                    {!isRecording && (
                      <button className="button small-button record-button" onClick={handleRecordStart}>
                        <FontAwesomeIcon icon={faMicrophone} />
                        Kayıt Başlat
                      </button>
                    )}
                    {isRecording && (
                      <button className="button small-button stop-button recording" onClick={handleRecordStop}>
                        <FontAwesomeIcon icon={faStop} />
                        Kayıt Durdur
                      </button>
                    )}
                  </div>

                  <input type="file" id="audio-upload-popup" accept="audio/*" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
                  <label htmlFor="audio-upload-popup" className="button">Dosya Yükle</label>
                </div>

                {isRecording && audioData && <AudioVisualizer audioData={audioData} />}
                
                {selectedFiles.length > 0 && (
                  <div className="file-info">
                    🎵 Seçilen Dosyalar: <strong>{selectedFiles.map(f => f.name).join(', ')}</strong>
                  </div>
                )}
              </div>
              
              <div className="popup-footer">
                <button className="button" onClick={() => setShowPopup(false)}>
                  İptal
                </button>
                <button 
                  className="button emotion-button" 
                  onClick={handleSave}
                  disabled={!formData.fullName || !formData.phoneNumber || selectedFiles.length === 0}
                >
                  <FontAwesomeIcon icon={faUserPlus} style={{ marginRight: '8px' }} />
                  Kaydet
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ManagementPage;