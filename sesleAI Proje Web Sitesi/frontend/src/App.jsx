//App.jsx
import React, { useState } from 'react';
import './App.css';
import VerificationPage from './VerificationPage.jsx';
import ManagementPage from './ManagementPage.jsx';
import EmotionVerificationPage from './EmotionVerificationPage.jsx';

const App = () => {
  const [currentPage, setCurrentPage] = useState('verify');
  const [verification, setVerification] = useState(null);

  const handleVerified = (verificationResult) => {
    setVerification(verificationResult);
    setCurrentPage('manage');
  };

  const handleBackToVerify = () => {
    setCurrentPage('verify');
    setVerification(null);
  };

  const openEmotionPage = () => {
    setCurrentPage('emotion');
  };

  const backToManage = () => {
    setCurrentPage('manage');
  };
  
  return (
    <div className="App">
      {currentPage === 'verify' && (
        <VerificationPage onVerified={handleVerified} />
      )}
      {currentPage === 'manage' && (
        <ManagementPage verification={verification} onBack={handleBackToVerify} onOpenEmotion={openEmotionPage} />
      )}
      {currentPage === 'emotion' && (
        <EmotionVerificationPage verification={verification} onBack={backToManage} />
      )}
    </div>
  );
};

export default App;
