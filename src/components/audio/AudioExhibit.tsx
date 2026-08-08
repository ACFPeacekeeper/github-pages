'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';

export function AudioExhibit() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaElementAudioSourceNode | null>(null);
  const requestRef = useRef<number>();
  
  // Clean up function to fully release AudioContext resources
  const cleanupAudio = useCallback(() => {
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return cleanupAudio;
  }, [cleanupAudio]);

  const initAudio = useCallback(() => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        audioContextRef.current = new AudioContextClass();
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        
        if (audioRef.current) {
          sourceNodeRef.current = audioContextRef.current.createMediaElementSource(audioRef.current);
          sourceNodeRef.current.connect(analyserRef.current);
          analyserRef.current.connect(audioContextRef.current.destination);
        }
      }
    }
    if (audioContextRef.current?.state === 'suspended') {
      audioContextRef.current.resume();
    }
  }, []);

  const draw = useCallback(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    requestRef.current = requestAnimationFrame(draw);
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    const bufferLength = analyserRef.current?.frequencyBinCount || 128;
    const dataArray = new Uint8Array(bufferLength);
    
    if (isDemoMode) {
      // Generate synthetic FFT data for non-audio demo mode
      const time = Date.now() / 1000;
      for (let i = 0; i < bufferLength; i++) {
        const val = Math.sin(time * 3 + i * 0.1) * Math.cos(time * 2 + i * 0.05);
        dataArray[i] = Math.max(0, 128 + 127 * val);
      }
    } else if (analyserRef.current && isPlaying) {
      analyserRef.current.getByteFrequencyData(dataArray);
    }
    
    // Render FFT data
    const barWidth = (width / bufferLength) * 2.5;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * height;
      ctx.fillStyle = `hsl(${(i / bufferLength) * 360}, 80%, 60%)`;
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }, [isDemoMode, isPlaying]);

  useEffect(() => {
    if (hasStarted) {
      if (!isDemoMode && isPlaying && audioRef.current) {
        audioRef.current.play().catch(console.error);
      } else if (audioRef.current) {
        audioRef.current.pause();
      }
      
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      requestRef.current = requestAnimationFrame(draw);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isPlaying, isDemoMode, hasStarted, draw]);

  const handlePlayPause = () => {
    if (!hasStarted) {
      // Audio starts ONLY after a user gesture
      initAudio();
      setHasStarted(true);
    }
    setIsPlaying(!isPlaying);
  };

  const handleDemoToggle = () => {
    setIsDemoMode(!isDemoMode);
    if (!hasStarted) {
      setHasStarted(true);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && audioRef.current) {
      const url = URL.createObjectURL(file);
      audioRef.current.src = url;
      setIsDemoMode(false);
      setIsPlaying(false);
      if (hasStarted) {
        setHasStarted(false); // Reset to ensure new user gesture if needed, or allow continuing
      }
    }
  };

  return (
    <div className="flex flex-col items-center gap-6 p-6 border border-gray-800 rounded-xl bg-gray-950 text-white shadow-2xl max-w-3xl mx-auto">
      <div className="text-center">
        <h2 className="text-2xl font-bold tracking-tight mb-2">Audio Reactive Signal</h2>
        <p className="text-gray-400 text-sm max-w-md mx-auto">
          Visualize audio frequencies using the Web Audio API. Upload local media or try the synthetic non-audio demo.
        </p>
      </div>

      <canvas
        ref={canvasRef}
        width={800}
        height={250}
        className="w-full max-w-full bg-black rounded-lg border border-gray-800 shadow-inner"
        aria-label="Audio frequency visualization canvas"
      />
      
      <div className="flex flex-wrap gap-4 items-center justify-center">
        <button
          onClick={handlePlayPause}
          className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition-all shadow-md focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 focus:ring-offset-gray-950 focus:outline-none"
          aria-label={isPlaying ? "Pause audio" : "Play audio"}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        
        <button
          onClick={handleDemoToggle}
          className={`px-6 py-2.5 rounded-lg font-medium transition-all shadow-md focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-gray-950 focus:outline-none ${isDemoMode ? 'bg-purple-600 hover:bg-purple-700 text-white' : 'bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700'}`}
          aria-pressed={isDemoMode}
        >
          {isDemoMode ? "Demo Active" : "Demo Mode"}
        </button>

        <div className="relative">
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="hidden"
            id="audio-upload-input"
            aria-label="Select local audio file"
          />
          <label
            htmlFor="audio-upload-input"
            className="cursor-pointer px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-medium transition-all shadow-md inline-block focus-within:ring-2 focus-within:ring-emerald-400 focus-within:ring-offset-2 focus-within:ring-offset-gray-950"
          >
            Select Local Media
          </label>
        </div>
      </div>
      
      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        className="hidden"
        controls={false}
      />
    </div>
  );
}
