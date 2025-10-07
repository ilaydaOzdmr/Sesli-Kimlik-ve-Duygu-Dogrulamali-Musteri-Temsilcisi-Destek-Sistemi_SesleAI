import React, { useState, useRef } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer
} from "recharts";
// import logoImg from "../AMBLEM_SARI.jpeg";
import "./styles.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [emotionKey, setEmotionKey] = useState(null); // raw key from API: happy/angry/sad/...
  const [packageRotation, setPackageRotation] = useState(0);
  const [emotionRotation, setEmotionRotation] = useState({});
  const [listening, setListening] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [chartData, setChartData] = useState([
    { name: "Wav2Vec2", value: 90, label: "Wav2Vec2 Model" },
  ]);

  const [speechText, setSpeechText] = useState("");
  const [listeningSpeech, setListeningSpeech] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  // Turkish -> English label fallback for status bar
  const trToEn = {
    "Mutlu": "happy",
    "Kızgın": "angry",
    "Şaşkın": "surprise",
    "Endişeli": "fear",
    "Hoşnutsuz": "disgust",
    "Üzgün": "sad",
    "Sakin": "calm",
    "Nötr": "neutral"
  };

  // English -> Turkish mapping to enforce Turkish everywhere
  const enToTr = {
    neutral: "Sakin",
    calm: "Sakin",
    happy: "Mutlu",
    sad: "Üzgün",
    angry: "Kızgın",
    fear: "Endişeli",
    disgust: "Hoşnutsuz",
    surprise: "Şaşkın"
  };

  // Robust translator that accepts TR/EN, normalizes and returns Turkish
  const translateToTr = (label) => {
    if (!label || typeof label !== 'string') return null;
    const key = label.toString().trim().toLowerCase();
    // direct english
    if (enToTr[key]) return enToTr[key];
    // common turkish variants to canonical
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

  // Customer profile sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Random helpers
  const getRandomFrom = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const getRandomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

  // Randomize current customer's package and usages
  const [currentPackage] = useState(() => {
    const internetOptions = [30, 40, 50];
    const total = getRandomFrom(internetOptions);
    const used = getRandomInt(Math.floor(total * 0.2), Math.floor(total * 0.6));
    return {
      name: `Platinum ${total}GB`,
      price: `${199 + (total - 30) * 20} TL`,
      internetTotal: total,
      internetUsed: used,
      minutesTotal: 100,
      minutesUsed: getRandomInt(10, 60),
    smsTotal: 1000,
      smsUsed: getRandomInt(100, 400)
    };
  });

  // Random customer profile data
  const [customerProfile] = useState(() => {
    const names = ["Ahmet Yılmaz", "Ayşe Demir", "Mehmet Kaya", "Zeynep Çelik", "Ali Kurt", "Elif Aksoy"]; 
    const fullName = getRandomFrom(names);
    const phoneNumber = `+90 5${getRandomInt(10,39)} ${getRandomInt(100,999)} ${getRandomInt(10,99)} ${getRandomInt(10,99)}`;
    const customerId = `TC${getRandomInt(100000000, 999999999)}`;
    const emailSlug = fullName.toLowerCase().replace(/[^a-zçğıöşü\s]/g, '').replace(/\s+/g, '.');
    const email = `${emailSlug}@mail.com`;
    const subscriptionType = getRandomFrom(["Bireysel", "Kurumsal"]);
    const packageStatus = getRandomFrom(["Aktif", "Yakında Yenileme", "Dondurulmuş"]);
    return { fullName, phoneNumber, customerId, email, subscriptionType, packageStatus };
  });

  // --- Canlı mikrofon ---
  const handleListen = async () => {
    if (listening) {
      mediaRecorderRef.current.stop();
      setListening(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);

      mediaRecorder.onstop = async () => {
        if (!audioChunksRef.current.length) return;
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });

        // Set local playback URL for recorded audio
        try {
          const url = URL.createObjectURL(audioBlob);
          if (audioUrl) URL.revokeObjectURL(audioUrl);
          setAudioUrl(url);
        } catch (_) {}

        const formData = new FormData();
        formData.append("file", audioBlob, "live_audio.wav");

        try {
          const res = await axios.post(`http://127.0.0.1:8001/predict`, formData, {
            headers: { "Content-Type": "multipart/form-data" }
          });

          const api = res.data || {};
          if (api && api.prediction) {
            setEmotionKey(api.prediction);
          }
          // Always normalize to Turkish
          const predTr = translateToTr(api.prediction_tr) || translateToTr(api.prediction);
          const newResults = api.prediction ? [{ model: 'Wav2Vec2', prediction: predTr, confidence: api.confidence || 0 }] : [];
          setResults(newResults);

          if (newResults.length) {
            const top = newResults.reduce(
              (a, b) => (a.confidence > b.confidence ? a : b),
              { confidence: 0 }
            );
            const nextEmotion = top.prediction || predTr || null;
            setCurrentEmotion(nextEmotion);
            setEmotionRotation((r) => ({ ...r, [nextEmotion]: ((r[nextEmotion] || 0) + 1) }));
          } else {
            const nextEmotion = predTr || null;
            setCurrentEmotion(nextEmotion);
            if (nextEmotion) {
              setEmotionRotation((r) => ({ ...r, [nextEmotion]: ((r[nextEmotion] || 0) + 1) }));
            }
          }
        } catch (err) {
          console.error(err);
          alert("Canlı analiz sırasında hata oluştu.");
        }
      };

      mediaRecorder.start();
      setListening(true);
    } catch (err) {
      console.error(err);
      alert("Mikrofon açılamadı.");
    }
  };

  // --- Dosya yükle ---
  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);

    // Set local playback URL for uploaded file
    try {
      const url = URL.createObjectURL(selectedFile);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(url);
    } catch (_) {}

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await axios.post(`http://127.0.0.1:8001/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      const api = res.data || {};
      if (api && api.prediction) {
        setEmotionKey(api.prediction);
      }
      const predTr = translateToTr(api.prediction_tr) || translateToTr(api.prediction);
      const newResults = api.prediction ? [{ model: 'Wav2Vec2', prediction: predTr, confidence: api.confidence || 0 }] : [];
      setResults(newResults);

      if (newResults.length) {
        const top = newResults.reduce(
          (a, b) => (a.confidence > b.confidence ? a : b),
          { confidence: 0 }
        );
        const nextEmotion = top.prediction || predTr || null;
        setCurrentEmotion(nextEmotion);
        setEmotionRotation((r) => ({ ...r, [nextEmotion]: ((r[nextEmotion] || 0) + 1) }));
      } else {
        const nextEmotion = predTr || null;
        setCurrentEmotion(nextEmotion);
        if (nextEmotion) {
          setEmotionRotation((r) => ({ ...r, [nextEmotion]: ((r[nextEmotion] || 0) + 1) }));
        }
      }
    } catch (err) {
      console.error(err);
      alert("Dosya yüklenirken hata oluştu.");
    }
  };

  // --- Konuşma Analizi ---
  const handleSpeechAnalysis = () => {
    setSpeechText("");
    setListeningSpeech(true);

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Tarayıcınız konuşma tanımayı desteklemiyor. Chrome, Edge veya Safari kullanın.");
      setListeningSpeech(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "tr-TR";
    recognition.interimResults = true;
    recognition.maxAlternatives = 3;
    recognition.continuous = false;

    let finalTranscript = "";
    let timeoutId;

    // 10 saniye timeout
    timeoutId = setTimeout(() => {
      if (listeningSpeech) {
        setSpeechText("Konuşma algılanamadı. Lütfen daha yüksek sesle konuşun ve tekrar deneyin.");
        setListeningSpeech(false);
        recognition.stop();
      }
    }, 10000);

    recognition.onstart = () => {
      console.log("Konuşma tanıma başladı...");
      setSpeechText("🎤 Dinleniyor... Lütfen konuşun.");
    };

    recognition.onresult = (event) => {
      let interimTranscript = "";
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      // Geçici sonuçları göster
      if (interimTranscript) {
        setSpeechText(`🎤 Dinleniyor: ${interimTranscript}`);
      }

      // Final sonucu göster
      if (finalTranscript) {
        clearTimeout(timeoutId);
        setSpeechText(finalTranscript);
        setListeningSpeech(false);
        console.log("Final transcript:", finalTranscript);
      }
    };

    recognition.onerror = (event) => {
      clearTimeout(timeoutId);
      console.error("Konuşma tanıma hatası:", event.error);
      let errorMessage = "Konuşma algılanamadı.";
      
      switch (event.error) {
        case "no-speech":
          errorMessage = "Konuşma algılanamadı. Lütfen daha yüksek sesle konuşun.";
          break;
        case "audio-capture":
          errorMessage = "Mikrofon erişimi reddedildi. Lütfen mikrofon iznini verin.";
          break;
        case "not-allowed":
          errorMessage = "Mikrofon izni verilmedi. Lütfen tarayıcı ayarlarından mikrofon iznini açın.";
          break;
        case "network":
          errorMessage = "Ağ hatası. İnternet bağlantınızı kontrol edin.";
          break;
        case "aborted":
          errorMessage = "Konuşma tanıma iptal edildi.";
          break;
        default:
          errorMessage = `Konuşma tanıma hatası: ${event.error}`;
      }
      
      setSpeechText(errorMessage);
      setListeningSpeech(false);
    };

    recognition.onend = () => {
      clearTimeout(timeoutId);
      console.log("Konuşma tanıma sonlandı.");
      setListeningSpeech(false);
    };

    try {
      recognition.start();
    } catch (error) {
      clearTimeout(timeoutId);
      console.error("Konuşma tanıma başlatılamadı:", error);
      setSpeechText("Konuşma tanıma başlatılamadı. Lütfen sayfayı yenileyin.");
      setListeningSpeech(false);
    }
  };

  const dynamicChartData = Array.isArray(results)
    ? results.map((r) => ({
        name: r.model,
        value: r.confidence * 100,
        label: r.prediction
      }))
    : chartData;

  // Usage chart for current package (remaining %)
  const usageChartData = (() => {
    const internetRemain = Math.max(currentPackage.internetTotal - currentPackage.internetUsed, 0);
    const minutesRemain = Math.max(currentPackage.minutesTotal - currentPackage.minutesUsed, 0);
    const smsRemain = Math.max(currentPackage.smsTotal - currentPackage.smsUsed, 0);
    return [
      {
        name: "İnternet",
        value: (internetRemain / Math.max(currentPackage.internetTotal, 1)) * 100,
        label: `${internetRemain} GB / ${currentPackage.internetTotal} GB`
      },
      {
        name: "Dakika",
        value: (minutesRemain / Math.max(currentPackage.minutesTotal, 1)) * 100,
        label: `${minutesRemain} dk / ${currentPackage.minutesTotal} dk`
      },
      {
        name: "SMS",
        value: (smsRemain / Math.max(currentPackage.smsTotal, 1)) * 100,
        label: `${smsRemain} / ${currentPackage.smsTotal}`
      }
    ];
  })();

  // Simplified recommendation logic based on Turkish emotions
  const getRecommendations = () => {
    const internet = currentPackage.internetTotal;
    const lower = Math.max(internet - 10, 5);
    const same = internet;
    const higher = internet + 10;

    const buildList = (type) => {
      switch (type) {
        case "upper":
        return [
            { title: `${higher}GB Plus`, price: `${229 + (higher - 30) * 2} TL`, desc: `${higher}GB + 1500 dk + 1500 SMS`, badge: "Üst", color: "#3b82f6" },
            { title: `${internet}GB Premium`, price: `${219 + (internet - 30) * 2} TL`, desc: `${internet}GB + 1000 dk + 1000 SMS`, badge: "Mevcut+", color: "#10b981" },
            { title: `Sosyal 10GB`, price: "99 TL", desc: "Sosyal medya ek 10GB", badge: "Ek", color: "#f59e0b" }
          ];
        case "lower":
        return [
            { title: `${lower}GB Ekonomik`, price: `${159 + (lower - 10)} TL`, desc: `Uygun fiyatlı ${lower}GB`, badge: "Alt", color: "#10b981" },
            { title: `Günlük 1GB`, price: "9 TL", desc: "Kullanım kadar öde", badge: "Günlük", color: "#f59e0b" },
            { title: `${same}GB Standart`, price: `${199 + (same - 30) * 2} TL`, desc: `${same}GB + 1000 dk + 1000 SMS`, badge: "Aynı", color: "#6b7280" }
          ];
        case "similar":
        return [
            { title: `${same}GB Standart`, price: `${199 + (same - 30) * 2} TL`, desc: `${same}GB + 1000 dk + 1000 SMS`, badge: "Benzer", color: "#6b7280" },
            { title: `${same}GB Esnek`, price: `${209 + (same - 30) * 2} TL`, desc: `${same}GB + Esnek kullanım`, badge: "Esnek", color: "#3b82f6" },
            { title: `Sosyal 10GB`, price: "99 TL", desc: "Sosyal medya 10GB", badge: "Ek", color: "#8b5cf6" }
          ];
        case "same_or_lower":
        return [
            { title: `${same}GB Mevcut`, price: `${199 + (same - 30) * 2} TL`, desc: `${same}GB + 1000 dk + 1000 SMS`, badge: "Mevcut", color: "#6b7280" },
            { title: `${lower}GB Ekonomik`, price: `${159 + (lower - 10)} TL`, desc: `Uygun fiyatlı ${lower}GB`, badge: "Alt", color: "#10b981" },
            { title: `Günlük 1GB`, price: "9 TL", desc: "Kullanım kadar öde", badge: "Günlük", color: "#f59e0b" }
          ];
        default:
          return [];
      }
    };

    let rule = "similar";
    switch (currentEmotion) {
      case "Mutlu":
        rule = "upper"; break;
      case "Üzgün":
      case "Hoşnutsuz":
      case "Korku":
        rule = "lower"; break;
      case "Sakin":
      case "Nötr":
        rule = "similar"; break;
      case "Kızgın":
        rule = "same_or_lower"; break;
      default:
        rule = "similar";
    }

    const list = buildList(rule);
    // Expand to 3-8 items by adding variants
    const extras = [
      { title: `${higher}GB Ultra`, price: `${259 + (higher - 30) * 3} TL`, desc: `${higher}GB + 2000 dk + 2000 SMS`, badge: "Üst", color: "#2563eb" },
      { title: `${lower}GB Mini`, price: `${139 + (lower - 10)} TL`, desc: `${lower}GB uygun fiyat`, badge: "Mini", color: "#10b981" },
      { title: `Haftalık 5GB`, price: "39 TL", desc: "Kısa dönem 5GB", badge: "Haftalık", color: "#06b6d4" }
    ];
    const combined = [...list, ...extras].slice(0, getRandomInt(3, 8));
    return combined;
  };

  const emojis = [
    { label: "Sakin", icon: "😐" },
    { label: "Mutlu", icon: "😊" },
    { label: "Üzgün", icon: "😢" },
    { label: "Kızgın", icon: "😡" },
    { label: "Hoşnutsuz", icon: "🤢" },
    { label: "Şaşkın", icon: "😮" },
    { label: "Endişeli", icon: "😨" }
  ];

  // Emotion color mapping
  const getEmotionColor = (emotion) => {
    const colors = {
      "Mutlu": "#10b981",
      "Üzgün": "#3b82f6", 
      "Kızgın": "#ef4444",
      "Endişeli": "#f59e0b",
      "Hoşnutsuz": "#8b5cf6",
      "Şaşkın": "#06b6d4",
      "Nötr": "#6b7280",
      "Sakin": "#6b7280"
    };
    return colors[emotion] || "#6b7280";
  };

  // Emotion icon mapping
  const getEmotionIcon = (emotion) => {
    const icons = {
      "Mutlu": "😊",
      "Üzgün": "😢", 
      "Kızgın": "😡",
      "Endişeli": "😨",
      "Hoşnutsuz": "🤢",
      "Şaşkın": "😮",
      "Nötr": "😐",
      "Sakin": "😐"
    };
    return icons[emotion] || "😐";
  };

  // Single source for UI: only show when there's a real prediction
  const activeEmotion = currentEmotion || translateToTr(emotionKey);

  return (
    <div className="app-container">
      {/* Customer Profile Sidebar */}
      <div className={`customer-sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h3>👤 Müşteri Profili</h3>
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? '✕' : '☰'}
          </button>
        </div>
        <div className="sidebar-content">
          <div className="profile-item">
            <span className="profile-label">Ad Soyad:</span>
            <span className="profile-value">{customerProfile.fullName}</span>
          </div>
          <div className="profile-item">
            <span className="profile-label">Telefon:</span>
            <span className="profile-value">{customerProfile.phoneNumber}</span>
          </div>
          <div className="profile-item">
            <span className="profile-label">Müşteri No:</span>
            <span className="profile-value">{customerProfile.customerId}</span>
          </div>
          <div className="profile-item">
            <span className="profile-label">E-posta:</span>
            <span className="profile-value">{customerProfile.email}</span>
          </div>
          <div className="profile-item">
            <span className="profile-label">Abonelik:</span>
            <span className="profile-value">{customerProfile.subscriptionType}</span>
          </div>
          <div className="profile-item">
            <span className="profile-label">Paket Durumu:</span>
            <span className="profile-value status-active">{customerProfile.packageStatus}</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className={`main-content ${sidebarOpen ? 'sidebar-open' : ''}`}>
    <div className="container">
      <div className="logo-banner">
        <div className="turkcell-logo">
              <img 
                className="logo-img"
                src="https://ffo3gv1cf3ir.merlincdn.net/SiteAssets/Hakkimizda/genel-bakis/logolarimiz/AMBLEM_SARI.png?20250928_03"
                alt="Turkcell"
                style={{ height: '48px', marginRight: '12px' }}
                onError={(e) => { e.currentTarget.src = '/AMBLEM_SARI.png'; }}
              />
        </div>
        <div className="subtitle">Müşteri Deneyimi Yönetimi & Duygu Analizi</div>
            <button 
              className="sidebar-toggle-main"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? "Müşteri profilini gizle" : "Müşteri profilini göster"}
            >
              {sidebarOpen ? '👤 Profil Açık' : '👤 Profil'}
            </button>
      </div>

      <div className="hero">
        <h1>Kurumsal Müşteri Yönetimi</h1>
        <p>Gelişmiş duygu analizi teknolojisi ile müşteri memnuniyetini optimize edin ve stratejik hizmet önerileri geliştirin.</p>

        <div className="buttons">
          <button className="yellow-btn primary-btn" onClick={handleListen}>
            {listening ? "⏹️ Durdur" : "🎤 Canlı Analiz"}
          </button>
          <label className="yellow-btn secondary-btn">
            📁 Dosya Yükle
            <input type="file" accept="audio/*" onChange={handleFileChange} hidden />
          </label>
          <button 
            className="yellow-btn tertiary-btn" 
            onClick={handleSpeechAnalysis}
            disabled={listeningSpeech}
            title={listeningSpeech ? "Konuşma tanıma devam ediyor..." : "Konuşma tanıma başlat"}
          >
            {listeningSpeech ? "🔊 İşleniyor..." : "🗣️ Konuşma Analizi"}
          </button>
        </div>

        {file && <p className="file-name">📁 Yüklenen Dosya: {file.name}</p>}

        {speechText && (
          <div className="speech-bubble">
            <h4>📊 Konuşma Analizi Sonucu:</h4>
            <p>{speechText}</p>
            {speechText.includes("Konuşma algılanamadı") || speechText.includes("hatası") || speechText.includes("izni") ? (
              <div className="speech-help">
                <p><strong>💡 İpuçları:</strong></p>
                <ul>
                  <li>Mikrofon iznini tarayıcı ayarlarından açın</li>
                  <li>Chrome, Edge veya Safari kullanın</li>
                  <li>Yüksek sesle ve net konuşun</li>
                  <li>Arka plan gürültüsünü azaltın</li>
                </ul>
              </div>
            ) : null}
          </div>
        )}

        {audioUrl && (
          <div className="audio-player">
            <audio controls src={audioUrl} />
          </div>
        )}

        <div className="emojis">
          {emojis.map((e) => (
            <div
              key={e.label}
              title={e.label}
              style={{
                border: e.label === activeEmotion ? "2px solid #007bff" : "none",
                borderRadius: "8px",
                padding: "6px 8px",
                display: "inline-flex",     // içerikler yan yana
                flexDirection: "row",       // emoji + yazı yatay
                alignItems: "center",       // dikeyde ortala
                gap: "4px",                 // emoji ile yazı arasına boşluk
                width: "auto"               // içerik kadar genişlik
                
              }}
              aria-label={e.label}
            >
              <div style={{ fontSize: "1.4rem" }}>{e.icon}</div>
              <div style={{ marginTop: "6px", fontSize: "0.8rem", color: "#333" }}>
             {e.label}
</div>

            </div>
            
          ))}
        </div>

        {/* Emotion Status Bar - Below Emojis */}
        {activeEmotion && (
          <div 
            className="emotion-status-bar"
            style={{ backgroundColor: getEmotionColor(activeEmotion) }}
          >
            <div className="emotion-icon">{getEmotionIcon(activeEmotion)}</div>
            <div className="emotion-text">{activeEmotion}</div>
          </div>
        )}

        {/* Grafiği tek model olduğunda gizle */}
        {dynamicChartData.length > 1 && (
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dynamicChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip
                  formatter={(value, name, props) => [
                    `${value.toFixed(1)}%`,
                    props.payload.label
                  ]}
                />
                <Bar dataKey="value" fill="#FFD700" stroke="blue" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Müşteri Mevcut Paket Bilgisi - Turkcell App Style */}
        <div className="current-package-section">
          <h2 style={{color:'#ffd700'}}>Müşteri Mevcut Paket Bilgisi</h2>
          <div className="current-package-grid">
            <div className="current-card">
              <div className="package-title"> {currentPackage.name}</div>
              <div className="package-price"> {currentPackage.price}</div>
              <div className="package-desc"> 🌐 {currentPackage.internetTotal}GB internet,📞{currentPackage.minutesTotal} dk, 💬 {currentPackage.smsTotal} SMS</div>
            </div>
            <div className="turkcell-usage-display">
              <h3> Kullanım Durumu</h3>
              <div className="usage-bars">
                <div className="usage-item">
                  <div className="usage-label">
                    <span>🌐 İnternet</span>
                    <span className="usage-text">{currentPackage.internetUsed}GB / {currentPackage.internetTotal}GB</span>
                  </div>
                  <div className="usage-bar-container">
                    <div 
                      className="usage-bar internet-bar"
                      style={{ 
                        width: `${(currentPackage.internetUsed / currentPackage.internetTotal) * 100}%`,
                        backgroundColor: '#10b981'
                      }}
                    ></div>
                  </div>
                </div>
                <div className="usage-item">
                  <div className="usage-label">
                    <span>📞 Dakika</span>
                    <span className="usage-text">{currentPackage.minutesUsed}dk / {currentPackage.minutesTotal}dk</span>
                  </div>
                  <div className="usage-bar-container">
                    <div 
                      className="usage-bar minutes-bar"
                      style={{ 
                        width: `${(currentPackage.minutesUsed / currentPackage.minutesTotal) * 100}%`,
                        backgroundColor: '#3b82f6'
                      }}
                    ></div>
                  </div>
                </div>
                <div className="usage-item">
                  <div className="usage-label">
                    <span>💬 SMS</span>
                    <span className="usage-text">{currentPackage.smsUsed} / {currentPackage.smsTotal}</span>
                  </div>
                  <div className="usage-bar-container">
                    <div 
                      className="usage-bar sms-bar"
                      style={{ 
                        width: `${(currentPackage.smsUsed / currentPackage.smsTotal) * 100}%`,
                        backgroundColor: '#8b5cf6'
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {currentEmotion && (
          <div className="packages-section">
            <h2 style={{color:'#ffd700'}}> Kurumsal Öneriler</h2>
            <p className="recommendation-subtitle">Müşterinin duygu durumuna göre stratejik hizmet önerileri:</p>
            {(() => {
              const list = getRecommendations();
              const rot = emotionRotation[currentEmotion] || 0;
              const rotateBy = list.length ? (rot % list.length) : 0;
              // Show 3-8 packages randomly
              const numPackages = Math.min(Math.max(3, Math.floor(Math.random() * 6) + 3), list.length);
              const cards = list.length ? [...list.slice(rotateBy), ...list.slice(0, rotateBy)].slice(0, numPackages) : [];
              return (
                <div className="packages-grid">
                  {cards.map((c, index) => (
                    <div 
                      key={`${c.title}-${index}`} 
                      className="package-card"
                      style={{ 
                        '--package-color': c.color || '#6b7280',
                        animationDelay: `${index * 0.1}s`
                      }}
                    >
                      <div 
                        className="package-badge"
                        style={{ backgroundColor: c.color || '#6b7280' }}
                      >
                        {c.badge}
                      </div>
                      <div className="package-title">{c.title}</div>
                      <div className="package-price">{c.price}</div>
                      <div className="package-desc">{c.desc}</div>
                      <button 
                        className="package-cta"
                        style={{ 
                          background: `linear-gradient(135deg, ${c.color || '#6b7280'}, ${c.color || '#6b7280'}dd)`
                        }}
                      >
                         Öner
                      </button>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        )}

        </div> {/* Close hero div */}

        <style jsx>{`
          .speech-bubble {
            margin-top: 20px;
            padding: 15px 20px;
            background: #f1f1f1;
            border-radius: 20px;
            max-width: 80%;
            position: relative;
            font-size: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          }
          .speech-bubble h4 {
            margin: 0 0 8px 0;
            font-weight: bold;
            color: #444;
          }
          .speech-bubble p {
            margin: 0;
            color: #222;
          }
          .speech-bubble::after {
            content: "";
            position: absolute;
            bottom: -15px;
            left: 40px;
            border-width: 15px 15px 0;
            border-style: solid;
            border-color: #f1f1f1 transparent transparent transparent;
          }
          .audio-player {
            margin-top: 12px;
          }
          /* Güncel Paket */
          .current-package-section {
            margin-top: 28px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 20px;
          }
          .current-package-grid {
            margin-top: 12px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            align-items: stretch;
          }
          .current-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.06);
          }
          .usage-chart {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 8px 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.06);
          }
          /* Turkcell Kurumsal Paket Tasarımı */
          .packages-section {
            margin-top: 28px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 20px;
          }
          .packages-grid {
            margin-top: 12px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
          }
          .package-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.06);
            transition: transform .15s ease, box-shadow .15s ease;
          }
          .package-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.08);
          }
          .package-badge {
            display: inline-block;
            background: #FFD700;
            color: #003366;
            font-weight: 700;
            font-size: 12px;
            padding: 6px 10px;
            border-radius: 999px;
          }
          .package-title {
            margin-top: 10px;
            font-size: 18px;
            font-weight: 700;
            color: #003366;
          }
          .package-price {
            margin-top: 6px;
            font-size: 16px;
            font-weight: 700;
            color: #003366;
          }
          .package-desc {
            margin-top: 6px;
            font-size: 14px;
            color: #334155;
          }
          .package-cta {
            margin-top: 12px;
            background: #003366;
            color: #ffffff;
            border: none;
            padding: 10px 14px;
            border-radius: 10px;
            cursor: pointer;
          }
          .package-cta:hover {
            background: #00264d;
          }
        `}</style>
        </div>
      </div>
    </div>
  );
}

