import React, { useState, useRef } from 'react';
import './App.css';
import toWav from 'audiobuffer-to-wav';
import AudioVisualizer from './AudioVisualizer.jsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone, faStop, faUserPlus, faUserCheck, faArrowLeft, faCheck, faTimes } from '@fortawesome/free-solid-svg-icons';
import logo from './DualMind_Logo.png';

const ManagementPage = ({ verification, onBack }) => {
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
    { basePhone: '05329991122', role: 'user',  fullName: 'Burak Acar',     packageName: 'GNC 8 GB' },
  ].map(s => ({
    ...s,
    suffix: s.role === 'owner' ? 'HatSahibi' : 'Kullanici',
    combinedName: `${s.basePhone}${s.role === 'owner' ? 'HatSahibi' : 'Kullanici'}`,
  }));

  const [status, setStatus] = useState('Hazır');
  const [result, setResult] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const [audioData, setAudioData] = useState(null);
  const [speakerName, setSpeakerName] = useState(verification?.inputName || '');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [role, setRole] = useState('owner'); // 'owner' (Hat Sahibi) | 'user' (Kullanıcı)
  const API_URL = 'http://localhost:8000';

  // Determine current profile from verification result (preferred) or current inputs
  const currentCombined = verification?.expectedName
    || (speakerName ? `${speakerName}${role === 'owner' ? 'HatSahibi' : 'Kullanici'}` : null);
  const currentProfile = currentCombined
    ? SEED_SUBSCRIBERS.find(s => s.combinedName === currentCombined)
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
        stream.getTracks().forEach(track => track.stop());
        setAudioData(null);
        setSelectedFiles(prev => [...prev, wavFile]);
        setStatus('Kayıt eklendi. Göndermek için Kayıt/Güncelle butonlarını kullanın.');
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

  const handleRegister = async () => {
    if (!speakerName || selectedFiles.length === 0) {
      setStatus('Lütfen bir isim girin ve ses kaydedin/yükleyin.');
      return;
    }
    const suffix = role === 'owner' ? 'HatSahibi' : 'Kullanici';
    const combinedName = `${speakerName}${suffix}`;
    const formData = new FormData();
    formData.append('name', combinedName);
    formData.append('role', role);
    selectedFiles.forEach((file, idx) => {
      const ext = (file.type && file.type.split('/')[1]) || 'wav';
      formData.append('audio_files', file, `${speakerName}_${idx}.${ext}`);
    });
    
    try {
      const response = await fetch(`${API_URL}/register/?name=${combinedName}&role=${role}`, { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        setStatus(`Konuşmacı '${combinedName}' başarıyla kaydedildi!`);
        setResult({ type: 'success', message: `Konuşmacı '${combinedName}' başarıyla kaydedildi!` });
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

  const handleUpdate = async () => {
    if (!speakerName || selectedFiles.length === 0) {
      setStatus('Lütfen bir isim girin ve ses kaydedin/yükleyin.');
      return;
    }
    const suffix = role === 'owner' ? 'HatSahibi' : 'Kullanici';
    const combinedName = `${speakerName}${suffix}`;
    const formData = new FormData();
    formData.append('name', combinedName);
    formData.append('role', role);
    selectedFiles.forEach((file, idx) => {
      const ext = (file.type && file.type.split('/')[1]) || 'wav';
      formData.append('audio_files', file, `${speakerName}_${idx}.${ext}`);
    });
    
    try {
      const response = await fetch(`${API_URL}/update_speaker/?name=${combinedName}&role=${role}`, { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        setStatus(`Konuşmacı '${combinedName}' başarıyla güncellendi!`);
        setResult({ type: 'success', message: `Konuşmacı '${combinedName}' başarıyla güncellendi!` });
      } else {
        setStatus('Güncelleme işlemi başarısız oldu.');
        setResult({ type: 'error', message: data.error || 'Bilinmeyen Hata' });
      }
    } catch (error) {
      console.error('Güncelleme API çağrısı sırasında hata:', error);
      setStatus('Sunucuya bağlanırken hata oluştu.');
      setResult({ type: 'error', message: 'Sunucuya bağlanırken hata oluştu.' });
    }
  };

  return (
    <div className="App">
      <div className="container">
        <img src={logo} alt="Sesli AI Logosu" className="logo" />
        <h1>Sonuç ve Konuşmacı Yönetimi</h1>

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

        <p className="subtitle">Kayıt veya güncelleme yapmak için aşağıdan ses yükleyin ya da kaydedin.</p>

        <div className="controls">
          <div className="speaker-input-container">
            <input type="text" placeholder="Konuşmacı adı" value={speakerName} onChange={(e) => setSpeakerName(e.target.value)} className="speaker-input" />
            <select className="role-select-small" value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="owner">Hat Sahibi</option>
              <option value="user">Kullanıcı</option>
            </select>
            {!isRecording && (
              <button className="button small-button record-button" onClick={handleRecordStart}>
                <FontAwesomeIcon icon={faMicrophone} />
              </button>
            )}
            {isRecording && (
              <button className="button small-button stop-button recording" onClick={handleRecordStop}>
                <FontAwesomeIcon icon={faStop} />
              </button>
            )}
          </div>

          <input type="file" id="audio-upload" accept="audio/*" multiple onChange={handleFileUpload} style={{ display: 'none' }} />
          <label htmlFor="audio-upload" className="button">Dosya Yükle</label>

          <div className="action-row">
            <button className="button" onClick={handleRegister} disabled={!speakerName || selectedFiles.length === 0}>
              <FontAwesomeIcon icon={faUserPlus} style={{ marginRight: '8px' }} />Yeni Kayıt
            </button>
            <button className="button emotion-button" onClick={handleUpdate} disabled={!speakerName || selectedFiles.length === 0}>
              <FontAwesomeIcon icon={faUserCheck} style={{ marginRight: '8px' }} />Mevcut Kaydı Güncelle
            </button>
          </div>

          <button className="button emotion-button" onClick={onBack}>
            <FontAwesomeIcon icon={faArrowLeft} style={{ marginRight: '8px' }} />Doğrulamaya Dön
          </button>
        </div>

        {isRecording && audioData && <AudioVisualizer audioData={audioData} />}
        <div className="status">Durum: <span>{status}</span></div>
        {selectedFiles.length > 0 && (
          <div className="file-info">
            🎵 Seçilen Dosyalar: <strong>{selectedFiles.map(f => f.name).join(', ')}</strong>
          </div>
        )}
        {result && (<div className={`result result-${result.type}`}>{result.message}</div>)}
      </div>
    </div>
  );
};

export default ManagementPage;

