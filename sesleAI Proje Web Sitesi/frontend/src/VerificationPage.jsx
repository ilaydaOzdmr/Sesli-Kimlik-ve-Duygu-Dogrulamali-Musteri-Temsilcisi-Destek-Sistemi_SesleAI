import React, { useState, useRef } from 'react';
import './App.css';
import toWav from 'audiobuffer-to-wav';
import AudioVisualizer from './AudioVisualizer.jsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone, faStop } from '@fortawesome/free-solid-svg-icons';
import logo from './DualMind_Logo.png';

const VerificationPage = ({ onVerified }) => {
  const [status, setStatus] = useState('Hazır');
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const [audioData, setAudioData] = useState(null);
  const [speakerName, setSpeakerName] = useState('');
  const [role, setRole] = useState('owner'); // 'owner' (Hat Sahibi) | 'user' (Kullanıcı)
  const API_URL = 'http://localhost:8000';

  const sendAudioToApi = async (audioBlob) => {
    if (!speakerName) {
      setStatus('Lütfen doğrulanacak kişinin adını girin.');
      return;
    }

    setStatus('Ses analizi için API\'ye gönderiliyor...');
    const formData = new FormData();
    const fileExtension = audioBlob.type.split('/')[1] || 'wav';
    formData.append('audio_file', audioBlob, `audio.${fileExtension}`);

    try {
      const response = await fetch(`${API_URL}/recognize/?role=${role}`, { method: 'POST', body: formData });
      const data = await response.json();
      const confidencePct = (data?.confidence ? (data.confidence * 100) : 0).toFixed(2);
      const suffix = role === 'owner' ? 'HatSahibi' : 'Kullanici';
      const expectedName = `${speakerName}${suffix}`;
      const isVerified = response.ok && data?.speaker === expectedName;
      onVerified({
        inputName: speakerName,
        apiSpeaker: data?.speaker || null,
        confidence: confidencePct,
        isVerified,
        message: isVerified ? `Doğrulandı (%${confidencePct})` : `Doğrulanamadı (%${confidencePct})`,
        role,
        expectedName
      });
    } catch (error) {
      console.error('API çağrısı sırasında hata:', error);
      const suffix = role === 'owner' ? 'HatSahibi' : 'Kullanici';
      const expectedName = `${speakerName}${suffix}`;
      onVerified({ inputName: speakerName, apiSpeaker: null, confidence: '0.00', isVerified: false, message: 'Sunucu hatası', role, expectedName });
    }
  };

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
        sendAudioToApi(wavFile);
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
    const file = event.target.files[0];
    if (file) {
      sendAudioToApi(file);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <img src={logo} alt="Sesli AI Logosu" className="logo" />
        <h1>Kimlik Doğrulama</h1>
        <p className="subtitle">Sadece isim girin ve ses yükleyin/çekin. Sonuç bir sonraki sayfada gösterilecek.</p>

        <div className="controls">
          <div className="speaker-input-container">
            <input
                type="text"
                placeholder="Doğrulanacak kişinin adını girin"
                value={speakerName}
                onChange={(e) => setSpeakerName(e.target.value)}
                className="speaker-input"
            />
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

          <input type="file" id="audio-upload" accept="audio/*" onChange={handleFileUpload} style={{ display: 'none' }} />
          <label htmlFor="audio-upload" className="button">Dosya Yükle</label>
        </div>

        {isRecording && audioData && <AudioVisualizer audioData={audioData} />}

        <div className="status">Durum: <span>{status}</span></div>
      </div>
    </div>
  );
};

export default VerificationPage;

