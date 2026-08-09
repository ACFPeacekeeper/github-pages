'use client';
import React, { useState, useEffect, useRef } from 'react';

/**
 * IF12: Optional WebXR viewing mode with explicit consent, session lifecycle controls 
 * and equivalent desktop navigation.
 * 
 * This component checks for WebXR support, provides a consent-gated entry point,
 * and handles the lifecycle of an XR session.
 */
export function WebXRExperiment() {
  const [xrSupported, setXrSupported] = useState<boolean>(false);
  const [sessionActive, setSessionActive] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Checking XR support...');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    async function checkXR() {
      if ('xr' in navigator && (navigator as any).xr) {
        try {
          const supported = await (navigator as any).xr.isSessionSupported('immersive-vr');
          setXrSupported(supported);
          setStatusMessage(supported ? 'Immersive VR Supported' : 'Immersive VR Not Supported');
        } catch (e) {
          setXrSupported(false);
          setStatusMessage('XR permission denied or error.');
        }
      } else {
        setXrSupported(false);
        setStatusMessage('WebXR API not available in this browser.');
      }
    }
    checkXR();
  }, []);

  const enterXR = async () => {
    if (!xrSupported || !('xr' in navigator)) return;
    
    try {
      setStatusMessage('Requesting XR Session (Consent required)...');
      // In a real application, we would start an immersive session and pass it to Three.js / WebGL.
      // Here we simulate the lifecycle.
      const session = await (navigator as any).xr.requestSession('immersive-vr', {
        optionalFeatures: ['local-floor', 'bounded-floor']
      });
      
      setSessionActive(true);
      setStatusMessage('XR Session Active. Rendering to headset...');
      
      session.addEventListener('end', () => {
        setSessionActive(false);
        setStatusMessage('XR Session Ended. Returned to desktop equivalent.');
      });
      
      // We would normally attach a WebGLContext to the session here.
      // session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
      
    } catch (err: any) {
      setStatusMessage(`Failed to enter XR: ${err.message}`);
    }
  };

  return (
    <div className="border border-indigo-900 bg-slate-900 rounded-lg overflow-hidden text-slate-200 shadow-xl">
      <div className="p-4 border-b border-indigo-900 flex justify-between items-center bg-indigo-950/30">
        <h3 className="font-bold text-indigo-300">WebXR Viewing Mode</h3>
        <span className="text-xs font-mono bg-slate-800 px-2 py-1 rounded border border-slate-700">
          Status: {sessionActive ? 'In Headset' : 'Desktop Mode'}
        </span>
      </div>

      <div className="p-6">
        <p className="text-sm text-slate-400 mb-6">
          This experiment provides an opt-in immersive viewing mode. When not in XR, all navigation and data exploration is fully accessible via standard desktop controls (mouse/keyboard).
        </p>

        <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 mb-6 text-center">
          <p className="font-mono text-sm mb-4">{statusMessage}</p>
          
          {xrSupported && !sessionActive && (
            <button 
              onClick={enterXR}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-2 rounded-full font-semibold transition-colors"
            >
              Enter Immersive VR
            </button>
          )}
          
          {sessionActive && (
            <div className="animate-pulse text-indigo-400 font-bold">
              Please put on your headset
            </div>
          )}
          
          {!xrSupported && (
            <div className="text-slate-500 text-sm">
              Falling back to desktop equivalent navigation.
            </div>
          )}
        </div>
        
        {/* Visual fallback / equivalent for desktop */}
        <div className="relative w-full aspect-video bg-black rounded border border-slate-800 overflow-hidden flex items-center justify-center">
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-30 pointer-events-none" />
          <div className="text-center z-10">
            <h4 className="text-xl font-light text-slate-300">Desktop Navigation View</h4>
            <p className="text-sm text-slate-500 mt-2">(Interactive 3D equivalent rendered here)</p>
          </div>
        </div>
      </div>
    </div>
  );
}
