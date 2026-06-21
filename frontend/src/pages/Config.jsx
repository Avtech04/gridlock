import { useState, useEffect } from 'react';
import axios from 'axios';
import { Settings, Key, CheckCircle, AlertCircle, ExternalLink } from 'lucide-react';

const API = 'http://localhost:8000';

export default function ConfigPage() {
  const [apiKey, setApiKey] = useState('');
  const [status, setStatus] = useState(null); // null | 'loading' | 'success' | 'error'
  const [isConnected, setIsConnected] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.get(`${API}/api/status`).then(r => setIsConnected(Boolean(r.data.engine_ready))).catch(() => {});
  }, []);

  const saveKey = async () => {
    if (!apiKey.trim()) return;
    setStatus('loading');
    try {
      const r = await axios.post(`${API}/api/config?api_key=${encodeURIComponent(apiKey.trim())}`);
      setStatus('success');
      setMessage(`${r.data.provider} vision review configured successfully.`);
      setIsConnected(true);
    } catch (e) {
      setStatus('error');
      setMessage(e.response?.data?.detail || 'Failed to configure API key.');
    }
  };

  return (
    <>
      <div className="page-header">
        <h2>System Configuration</h2>
        <p>Configure detector, optional vision review, and evaluation workflow</p>
      </div>

      <div className="page-content">
        <div style={{maxWidth:'700px'}}>
          {/* Current Status */}
          <div className="card" style={{marginBottom:'1.5rem'}}>
            <div className="card-header">
              <h3><Settings size={16} color="var(--accent-blue)" /> System Status</h3>
            </div>
            <div className="card-body">
              <div style={{display:'flex', alignItems:'center', gap:'12px', padding:'1rem', background: isConnected ? 'rgba(16,185,129,0.08)' : 'rgba(244,63,94,0.08)', border: `1px solid ${isConnected ? 'rgba(16,185,129,0.2)' : 'rgba(244,63,94,0.2)'}`, borderRadius:'10px'}}>
                {isConnected ? (
                  <CheckCircle size={24} color="var(--accent-emerald)" />
                ) : (
                  <AlertCircle size={24} color="var(--accent-rose)" />
                )}
                <div>
                  <div style={{fontWeight:600, fontSize:'0.95rem'}}>
                    {isConnected ? 'Detector Pipeline Ready' : 'Detector Pipeline Not Ready'}
                  </div>
                  <div style={{fontSize:'0.8rem', color:'var(--text-muted)', marginTop:'2px'}}>
                    {isConnected
                      ? 'The system is ready to analyze traffic images with full AI capabilities.'
                      : 'Install backend requirements and make sure detector weights are available.'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* API Key Config */}
          <div className="card">
            <div className="card-header">
              <h3><Key size={16} color="var(--accent-amber)" /> Optional Vision Review API Key</h3>
            </div>
            <div className="card-body">
              <p style={{fontSize:'0.85rem', color:'var(--text-secondary)', marginBottom:'1rem', lineHeight:'1.6'}}>
                The core system runs on detector-grounded CV. Groq or Google Gemini can be added as a high-confidence reviewer for ambiguous violations.
              </p>

              <div style={{marginBottom:'0.5rem'}}>
                <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer"
                  style={{display:'inline-flex', alignItems:'center', gap:'6px', color:'var(--accent-emerald)', fontSize:'0.85rem', textDecoration:'none', fontWeight:600}}>
                  <ExternalLink size={14} /> Get a FREE Groq API key (recommended) →
                </a>
              </div>
              <div style={{marginBottom:'1rem'}}>
                <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer"
                  style={{display:'inline-flex', alignItems:'center', gap:'6px', color:'var(--text-muted)', fontSize:'0.8rem', textDecoration:'none'}}>
                  <ExternalLink size={14} /> Or get a Gemini key from Google AI Studio
                </a>
              </div>

              <div style={{display:'flex', gap:'0.75rem'}}>
                <input
                  className="config-input"
                  type="password"
                  placeholder="Paste Groq (gsk_...) or Gemini (AIza...) key here..."
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && saveKey()}
                />
                <button className="btn btn-primary" onClick={saveKey} disabled={status === 'loading' || !apiKey.trim()}>
                  {status === 'loading' ? <div className="loader"></div> : 'Save'}
                </button>
              </div>

              {status === 'success' && (
                <div style={{marginTop:'1rem', padding:'0.75rem', background:'rgba(16,185,129,0.08)', border:'1px solid rgba(16,185,129,0.2)', borderRadius:'8px', fontSize:'0.85rem', color:'var(--accent-emerald)', display:'flex', alignItems:'center', gap:'8px'}}>
                  <CheckCircle size={16} /> {message}
                </div>
              )}
              {status === 'error' && (
                <div style={{marginTop:'1rem', padding:'0.75rem', background:'rgba(244,63,94,0.08)', border:'1px solid rgba(244,63,94,0.2)', borderRadius:'8px', fontSize:'0.85rem', color:'var(--accent-rose)', display:'flex', alignItems:'center', gap:'8px'}}>
                  <AlertCircle size={16} /> {message}
                </div>
              )}
            </div>
          </div>

          {/* Architecture Info */}
          <div className="card" style={{marginTop:'1.5rem'}}>
            <div className="card-header">
              <h3>System Architecture</h3>
            </div>
            <div className="card-body">
              <table className="data-table">
                <tbody>
                  <tr><td style={{fontWeight:600, width:'40%'}}>Primary Detector</td><td>Fine-tuned YOLO traffic model, YOLO26/YOLO11 fallback, local YOLOv3-tiny offline fallback</td></tr>
                  <tr><td style={{fontWeight:600}}>Specialist Evidence</td><td>Helmet, no-helmet, seatbelt, no-seatbelt, plate, stop-line, and traffic-light classes</td></tr>
                  <tr><td style={{fontWeight:600}}>Image Preprocessing</td><td>OpenCV illumination normalization, denoising, and deblur-safe sharpening</td></tr>
                  <tr><td style={{fontWeight:600}}>License Plates</td><td>Plate detector plus optional PaddleOCR, EasyOCR, or Tesseract OCR</td></tr>
                  <tr><td style={{fontWeight:600}}>Violation Logic</td><td>Detector-grounded rules with optional high-confidence vision-model review</td></tr>
                  <tr><td style={{fontWeight:600}}>Evaluation</td><td>Ground-truth JSON evaluation for Precision, Recall, F1-score, and mAP50</td></tr>
                  <tr><td style={{fontWeight:600}}>Backend</td><td>Python FastAPI v3.0</td></tr>
                  <tr><td style={{fontWeight:600}}>Frontend</td><td>React 19 + Vite + Recharts</td></tr>
                  <tr><td style={{fontWeight:600}}>Supported Violations</td><td>Helmet, Seatbelt, Triple Riding, Wrong-side, Stop-line, Red-light, Illegal Parking</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
