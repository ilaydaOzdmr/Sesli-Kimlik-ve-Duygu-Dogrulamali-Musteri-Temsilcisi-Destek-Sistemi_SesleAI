//App.jsx
import React, { useState } from 'react';
import './App.css';
import VerificationPage from './VerificationPage.jsx';
import ManagementPage from './ManagementPage.jsx';

const App = () => {
  const [currentPage, setCurrentPage] = useState('verify');
  const [verification, setVerification] = useState(null);

  const handleVerified = (verificationResult) => {
    setVerification(verificationResult);
    setCurrentPage('manage');
  };

  const handleBack = () => {
    setCurrentPage('verify');
    setVerification(null);
  };
  
  return (
    <div className="App">
      {currentPage === 'verify' && (
        <VerificationPage onVerified={handleVerified} />
      )}
      {currentPage === 'manage' && (
        <ManagementPage verification={verification} onBack={handleBack} />
      )}
    </div>
  );
};

export default App;
