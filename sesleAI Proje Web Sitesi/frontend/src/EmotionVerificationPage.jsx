import React, { useState, useRef, useMemo } from 'react';
import toWav from 'audiobuffer-to-wav';
import './App.css';
import logo from './DualMind_Logo.png';

const EmotionVerificationPage = ({ onBack, verification }) => {
  const [status, setStatus] = useState('Hazır');
  const [isListening, setIsListening] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [audioUrl, setAudioUrl] = useState(null);
  const [results, setResults] = useState([]); // [{model, prediction, confidence}]
  const [speechText, setSpeechText] = useState('');

  const API_URL = 'http://127.0.0.1:8001';

  // Debug: verification prop'unu kontrol et
  console.log('EmotionVerificationPage - verification:', verification);

  const translateToTr = (label) => {
    if (!label || typeof label !== 'string') return null;
    const key = label.toString().trim().toLowerCase();
    const enToTr = {
      neutral: 'Sakin',
      calm: 'Sakin',
      happy: 'Mutlu',
      sad: 'Üzgün',
      angry: 'Kızgın',
      fear: 'Endişeli',
      fearful: 'Endişeli',
      disgust: 'Hoşnutsuz',
      surprised: 'Şaşkın',
      surprise: 'Şaşkın'
    };
    if (enToTr[key]) return enToTr[key];
    const trMap = {
      'nötr': 'Sakin',
      'sakin': 'Sakin',
      'mutlu': 'Mutlu',
      'üzgün': 'Üzgün',
      'kızgın': 'Kızgın',
      'kizgin': 'Kızgın',
      'korku': 'Endişeli',
      'endişeli': 'Endişeli',
      'endiseli': 'Endişeli',
      'tiksinmiş': 'Hoşnutsuz',
      'hosnutsuz': 'Hoşnutsuz',
      'hoşnutsuz': 'Hoşnutsuz',
      'şaşkın': 'Şaşkın',
      'saskin': 'Şaşkın'
    };
    return trMap[key] || null;
  };

  // Seed subscribers
  const SEED_SUBSCRIBERS = useMemo(() => ([
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
  }))), []);

  const subscribers = useMemo(() => {
    try {
      const saved = localStorage.getItem('subscribers');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  }, []);

  const allSubscribers = useMemo(() => ([...SEED_SUBSCRIBERS, ...subscribers]), [SEED_SUBSCRIBERS, subscribers]);
  const currentCombined = verification?.expectedName;
  const currentProfile = useMemo(() => {
    if (currentCombined) {
      return allSubscribers.find(s => s.combinedName === currentCombined);
    }
    // Fallback: İlk kullanıcıyı al
    return allSubscribers[0] || null;
  }, [allSubscribers, currentCombined]);

  // Paket bilgileri - localStorage'dan gelen kullanıcıya göre
  const getPackageInfo = useMemo(() => {
    if (!currentProfile?.packageName) return null;
    
    const packageMap = {
      'GNC 30 GB': { internet: 30, minutes: 1000, sms: 1000, price: '₺89.90/ay' },
      'GNC 20 GB': { internet: 20, minutes: 1000, sms: 1000, price: '₺69.90/ay' },
      'GNC 8 GB': { internet: 8, minutes: 500, sms: 500, price: '₺49.90/ay' },
      'Platinum 40 GB': { internet: 40, minutes: 1500, sms: 1500, price: '₺119.90/ay' },
      'Super 10 GB': { internet: 10, minutes: 500, sms: 500, price: '₺59.90/ay' }
    };
    
    const baseInfo = packageMap[currentProfile.packageName] || { internet: 30, minutes: 1000, sms: 1000, price: '₺89.90/ay' };
    
    // Rastgele kullanım verileri
    const getRandomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    
    return {
      name: currentProfile.packageName,
      price: baseInfo.price,
      internetTotal: baseInfo.internet,
      internetUsed: getRandomInt(5, Math.floor(baseInfo.internet * 0.8)),
      minutesTotal: baseInfo.minutes,
      minutesUsed: getRandomInt(50, Math.floor(baseInfo.minutes * 0.7)),
      smsTotal: baseInfo.sms,
      smsUsed: getRandomInt(20, Math.floor(baseInfo.sms * 0.6))
    };
  }, [currentProfile]);

  // Emojis ve durum
  const emojis = [
    { label: 'Sakin', icon: '😐' },
    { label: 'Mutlu', icon: '😊' },
    { label: 'Üzgün', icon: '😢' },
    { label: 'Kızgın', icon: '😡' },
    { label: 'Hoşnutsuz', icon: '🤢' },
    { label: 'Şaşkın', icon: '😮' },
    { label: 'Endişeli', icon: '😨' }
  ];

  const topResult = Array.isArray(results) && results.length ? results[0] : null;
  const activeEmotion = topResult?.prediction || null;

  const getRecommendations = useMemo(() => {
    return () => {
      // Basit öneri listesi (duyguya göre çeşitlenir)
      const internet = 30;
      const same = internet;
      const lower = 20;
      const higher = 40;
      const base = [
        { title: `${same}GB Standart`, price: `${199} TL`, desc: `${same}GB + 1000 dk + 1000 SMS` },
        { title: `${higher}GB Plus`, price: `${229} TL`, desc: `${higher}GB + 1500 dk + 1500 SMS` },
        { title: `${lower}GB Ekonomik`, price: `${159} TL`, desc: `${lower}GB uygun fiyat` },
      ];
      switch (activeEmotion) {
        case 'Mutlu':
          return [base[1], base[0]];
        case 'Üzgün':
        case 'Hoşnutsuz':
        case 'Endişeli':
          return [base[2], base[0]];
        case 'Kızgın':
          return [base[0], base[2]];
        default:
          return base;
      }
    };
  }, [activeEmotion]);

  const startStopListening = async () => {
    if (isListening) {
      try { mediaRecorderRef.current && mediaRecorderRef.current.stop(); } catch (_) {}
      setIsListening(false);
      setStatus('Dinleme durduruldu.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/wav';
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);

      mediaRecorder.onstop = async () => {
        if (!audioChunksRef.current.length) return;
        const recordedBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType });
        try {
          const arrayBuffer = await recordedBlob.arrayBuffer();
          const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
          const wavBuffer = toWav(audioBuffer);
          const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
          try {
            const url = URL.createObjectURL(wavBlob);
            if (audioUrl) URL.revokeObjectURL(audioUrl);
            setAudioUrl(url);
          } catch (_) {}
          const formData = new FormData();
          formData.append('file', wavBlob, 'live_audio.wav');
          setStatus('Analiz ediliyor...');
          const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
          const api = await res.json();
          if (api && api.prediction) {
            const predTr = translateToTr(api.prediction_tr) || translateToTr(api.prediction) || api.prediction;
            const data = [{ model: 'Wav2Vec2', prediction: predTr, confidence: api.confidence || 0 }];
            setResults(data);
            setStatus('Analiz tamamlandı.');
          } else if (api && api.error) {
            setStatus(`Hata: ${api.error}`);
          } else {
            setStatus('Analiz sonucu alınamadı.');
          }
        } catch (err) {
          console.error(err);
          setStatus('Kayıt işlenirken hata oluştu.');
        } finally {
          stream.getTracks().forEach(t => t.stop());
        }
      };

      mediaRecorder.start();
      setIsListening(true);
      setStatus('Dinleniyor...');
    } catch (err) {
      console.error(err);
      setStatus('Mikrofon açılamadı.');
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      const url = URL.createObjectURL(file);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(url);
    } catch (_) {}

    const formData = new FormData();
    formData.append('file', file);
    try {
      setStatus('Analiz ediliyor...');
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      const api = await res.json();
      if (api && api.prediction) {
        const predTr = translateToTr(api.prediction_tr) || translateToTr(api.prediction) || api.prediction;
        const data = [{ model: 'Wav2Vec2', prediction: predTr, confidence: api.confidence || 0 }];
        setResults(data);
        setStatus('Analiz tamamlandı.');
      } else if (api && api.error) {
        setStatus(`Hata: ${api.error}`);
      } else {
        setStatus('Analiz sonucu alınamadı.');
      }
    } catch (err) {
      console.error(err);
      setStatus('Dosya yüklenirken hata oluştu.');
    }
  };

  const handleSpeechAnalysis = () => {
    setSpeechText('');
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechText('Tarayıcınız konuşma tanımayı desteklemiyor.');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'tr-TR';
    recognition.interimResults = true;
    recognition.maxAlternatives = 3;
    recognition.continuous = false;

    let finalTranscript = '';
    let timeoutId = setTimeout(() => {
      setSpeechText('Konuşma algılanamadı. Lütfen daha yüksek sesle konuşun ve tekrar deneyin.');
      try { recognition.stop(); } catch (_) {}
    }, 10000);

    recognition.onstart = () => { setSpeechText('🎤 Dinleniyor... Lütfen konuşun.'); };
    recognition.onresult = (event) => {
      let interimTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTranscript += transcript; else interimTranscript += transcript;
      }
      if (interimTranscript) setSpeechText(`🎤 Dinleniyor: ${interimTranscript}`);
      if (finalTranscript) { clearTimeout(timeoutId); setSpeechText(finalTranscript); }
    };
    recognition.onerror = (event) => { clearTimeout(timeoutId); setSpeechText(`Konuşma tanıma hatası: ${event.error}`); };
    recognition.onend = () => { clearTimeout(timeoutId); };
    try { recognition.start(); } catch (_) {}
  };

  // Debug: Sayfa yükleniyor mu kontrol et
  console.log('EmotionVerificationPage render ediliyor');

  return (
    <div className="App">
      {/* Modern Sidebar Toggle Button */}
      <button 
        className="modern-sidebar-toggle" 
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        <span className="toggle-icon">📊</span>
        <span className="toggle-text">Müşteri Detayları</span>
      </button>

      {/* Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="sidebar-overlay" 
          onClick={() => setSidebarOpen(false)}
        ></div>
      )}

      {/* Modern Sidebar */}
      <div className={`modern-sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="header-content">
            <div className="header-icon">👤</div>
            <h3>Müşteri Bilgileri</h3>
          </div>
          <button 
            className="close-btn" 
            onClick={() => setSidebarOpen(false)}
          >
            ✕
          </button>
        </div>
        
        <div className="sidebar-content">
          {/* Müşteri Bilgileri */}
          {currentProfile && (
            <div className="info-section">
              <h4 className="section-title">👤 Müşteri Bilgileri</h4>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">Ad Soyad</span>
                  <span className="info-value">{currentProfile.fullName}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Telefon</span>
                  <span className="info-value">{currentProfile.basePhone}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Rol</span>
                  <span className="info-value">{currentProfile.role === 'owner' ? 'Hat Sahibi' : 'Kullanıcı'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Paket Bilgileri */}
          {getPackageInfo && (
            <div className="info-section">
              <h4 className="section-title">📦 Mevcut Paket</h4>
              <div className="package-info">
                <div className="package-name">{getPackageInfo.name}</div>
                <div className="package-price">{getPackageInfo.price}</div>
                <div className="package-features">
                  🌐 {getPackageInfo.internetTotal}GB internet<br/>
                  📞 {getPackageInfo.minutesTotal} dk<br/>
                  💬 {getPackageInfo.smsTotal} SMS
                </div>
              </div>
            </div>
          )}

          {/* Kullanım Durumu */}
          {getPackageInfo && (
            <div className="info-section">
              <h4 className="section-title">📊 Kullanım Durumu</h4>
              <div className="usage-stats">
                <div className="usage-item">
                  <div className="usage-header">
                    <span className="usage-icon">🌐</span>
                    <span className="usage-label">İnternet</span>
                    <span className="usage-text">{getPackageInfo.internetUsed}GB / {getPackageInfo.internetTotal}GB</span>
                  </div>
                  <div className="usage-bar">
                    <div 
                      className="usage-fill internet"
                      style={{ width: `${(getPackageInfo.internetUsed / getPackageInfo.internetTotal) * 100}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="usage-item">
                  <div className="usage-header">
                    <span className="usage-icon">📞</span>
                    <span className="usage-label">Dakika</span>
                    <span className="usage-text">{getPackageInfo.minutesUsed}dk / {getPackageInfo.minutesTotal}dk</span>
                  </div>
                  <div className="usage-bar">
                    <div 
                      className="usage-fill minutes"
                      style={{ width: `${(getPackageInfo.minutesUsed / getPackageInfo.minutesTotal) * 100}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="usage-item">
                  <div className="usage-header">
                    <span className="usage-icon">💬</span>
                    <span className="usage-label">SMS</span>
                    <span className="usage-text">{getPackageInfo.smsUsed} / {getPackageInfo.smsTotal}</span>
                  </div>
                  <div className="usage-bar">
                    <div 
                      className="usage-fill sms"
                      style={{ width: `${(getPackageInfo.smsUsed / getPackageInfo.smsTotal) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Önerilen Paketler */}
          {activeEmotion && (
            <div className="info-section">
              <h4 className="section-title">💡 Önerilen Paketler</h4>
              <div className="recommendations">
                {getRecommendations().slice(0, 3).map((rec, idx) => (
                  <div key={`${rec.title}-${idx}`} className="recommendation-card">
                    <div className="rec-title">{rec.title}</div>
                    <div className="rec-price">{rec.price}</div>
                    <div className="rec-desc">{rec.desc}</div>
                    <button className="rec-button">Öner</button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="container">
        <img src={logo} alt="Sesli AI Logosu" className="logo" />
        <h1>Duygu Doğrulama</h1>
        <p className="subtitle">Sesinizi canlı kaydedin veya dosya yükleyin; duygu sonucu ve öneriler burada görünecek.</p>

        {currentProfile && (
          <div className="user-info">
            <span><strong>Ad Soyad:</strong> {currentProfile.fullName}</span>
            <span><strong>Telefon:</strong> {currentProfile.basePhone}</span>
            <span><strong>Paket:</strong> {currentProfile.packageName}</span>
            <span><strong>Rol:</strong> {currentProfile.role === 'owner' ? 'Hat Sahibi' : 'Kullanıcı'}</span>
          </div>
        )}

        <div className="controls">
          <button className="button" onClick={startStopListening}>{isListening ? '⏹️ Durdur' : '🎤 Canlı Analiz'}</button>
          <label htmlFor="emotion-file" className="button upload-button">📁 Dosya Yükle</label>
          <input id="emotion-file" type="file" accept="audio/*" onChange={handleFileChange} style={{ display: 'none' }} />
          <button className="button emotion-button" onClick={handleSpeechAnalysis}>🗣️ Konuşma Analizi</button>
          <button className="button emotion-button" onClick={onBack}>Geri</button>
        </div>

        {audioUrl && (
          <div style={{
            marginTop: 16,
            background: '#f5f7fb',
            border: '2px solid #1a237e',
            borderRadius: 12,
            boxShadow: '0 8px 24px rgba(13, 27, 111, 0.22)',
            padding: 8
          }}>
            <style>{`
              .custom-audio::-webkit-media-controls-panel { background-color: #f5f7fb; }
              .custom-audio::-webkit-media-controls-enclosure { background-color: #f5f7fb; border-radius: 12px; }
              .custom-audio::-webkit-media-controls-time-remaining-display,
              .custom-audio::-webkit-media-controls-current-time-display { color: #0f172a; }
              .custom-audio { accent-color: #1a237e; }
              .custom-audio::-webkit-media-controls-timeline-container { background-color: #f5f7fb !important; border-radius: 8px; }
              .custom-audio::-webkit-media-controls-timeline { background-color: #f5f7fb !important; border-radius: 8px; height: 6px; }
              .custom-audio::-webkit-media-controls-progress-bar { background-color:rgb(7, 37, 97) !important; border-radius: 8px; }
            `}</style>
            <audio
              className="custom-audio"
              controls
              src={audioUrl}
              style={{ width: '100%', height: 26, transform: 'scaleY(0.90)', transformOrigin: 'center' }}
            />
          </div>
        )}

        <div className="status">Durum: <span>{status}</span></div>

        {/* Emojiler */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 12, marginTop: 20 }}>
          {emojis.map((e) => (
            <div key={e.label} style={{ background: '#f8fafc', border: '1px solid #dbe2f3', borderRadius: 14, padding: 12, boxShadow: '0 6px 16px rgba(15,23,42,0.06)', outline: activeEmotion===e.label ? '2px solid #1a237e' : 'none' }}>
              <div style={{ fontSize: '1.6rem' }}>{e.icon}</div>
              <div style={{ marginTop: 6, fontWeight: 800, color: '#0f172a' }}>{e.label}</div>
            </div>
          ))}
        </div>

        {/* Aktif Duygu Kutu */}
        {activeEmotion && (
          <div style={{
            marginTop: 18,
            display: 'block',
            width: 'fit-content',
            padding: '16px 22px',
            background: '#ffffff',
            border: '8px solid #1a237e',
            borderRadius: 16,
            color: '#0f172a',
            fontSize: '1.3rem',
            fontWeight: 600,
            marginLeft: 'auto',
            marginRight: 'auto'
          }}>
            <span style={{ fontWeight: 900, marginRight: 8 }}>Aktif Duygu:</span>
            <span>{activeEmotion}</span>
          </div>
        )}

        {/* Öneriler */}
        {activeEmotion && (
          <div style={{ marginTop: 24, textAlign: 'left' }}>
            <h2 style={{ fontSize: '1.3rem', color: '#0f172a', marginBottom: 8 }}>Öneriler</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
              {getRecommendations().map((c, idx) => (
                <div key={`${c.title}-${idx}`} style={{ background: '#ffffff', border: '1px solid #e6eaf0', borderRadius: 14, padding: 12, boxShadow: '0 6px 16px rgba(15,23,42,0.06)' }}>
                  <div style={{ fontWeight: 700, color: '#0b144a' }}>{c.title}</div>
                  <div style={{ fontWeight: 800, color: '#ffbf00', marginTop: 4 }}>{c.price}</div>
                  <div style={{ color: '#334155', marginTop: 6 }}>{c.desc}</div>
                  <button className="button" style={{ marginTop: 10 }}>Öner</button>
                </div>
              ))}
            </div>
          </div>
        )}

        {speechText && (
          <div className="file-info" style={{ textAlign: 'left' }}>
            <strong>Konuşma Analizi:</strong> {speechText}
          </div>
        )}
      </div>
    </div>
  );
};

export default EmotionVerificationPage;
