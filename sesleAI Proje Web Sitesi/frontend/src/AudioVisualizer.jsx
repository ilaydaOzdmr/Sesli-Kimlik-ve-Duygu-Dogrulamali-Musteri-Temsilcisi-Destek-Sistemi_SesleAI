import React, { useRef, useEffect } from 'react';

const AudioVisualizer = ({ audioData }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!audioData) return;

    const canvas = canvasRef.current;
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioCtx.createAnalyser();
    
    const source = audioCtx.createMediaStreamSource(audioData);
    source.connect(analyser);

    analyser.fftSize = 2048;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const width = canvas.width;
    const height = canvas.height;
    const canvasCtx = canvas.getContext('2d');

    const draw = () => {
      requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      canvasCtx.fillStyle = '#1a237e';
      canvasCtx.fillRect(0, 0, width, height);
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = '#ffc107';
      canvasCtx.beginPath();
      let sliceWidth = width * 1.0 / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        let v = dataArray[i] / 128.0;
        let y = height / 2 + v * height / 4;
        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
    };

    draw();
    
    return () => {
      source.disconnect();
      analyser.disconnect();
    };
  }, [audioData]);

  return (
    <canvas ref={canvasRef} width={300} height={80} style={{ marginTop: '20px' }} />
  );
};

export default AudioVisualizer;