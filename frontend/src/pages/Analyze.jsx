import { useState, useRef } from 'react';
import axios from 'axios';
import {
  UploadCloud, Search, CheckCircle, AlertTriangle,
  Image as ImageIcon, Zap, Eye, FileText
} from 'lucide-react';

const API = 'http://localhost:8000';

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState('');
  const [results, setResults] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleFile = (f) => {
    if (!f.type.startsWith('image/')) return alert('Please upload an image.');
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResults(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setResults(null);

    const steps = ['Uploading evidence...', 'Enhancing image quality...', 'Running detector cascade...', 'Running OCR and rule engine...', 'Generating annotated evidence...'];
    let stepIdx = 0;
    setAnalysisStep(steps[0]);
    const stepInterval = setInterval(() => {
      stepIdx = Math.min(stepIdx + 1, steps.length - 1);
      setAnalysisStep(steps[stepIdx]);
    }, 1500);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (error) {
      const detail = error.response?.data?.detail || 'Analysis failed. Check backend.';
      alert(detail);
    } finally {
      clearInterval(stepInterval);
      setIsAnalyzing(false);
      setAnalysisStep('');
    }
  };

  const getConfClass = (conf) => conf >= 0.8 ? 'high' : conf >= 0.5 ? 'medium' : 'low';

  return (
    <>
      <div className="page-header">
        <h2>Analyze Image</h2>
        <p>Upload a traffic surveillance image for detector-grounded violation analysis</p>
      </div>

      <div className="page-content">
        <div className="grid-2">
          {/* Left: Upload & Preview */}
          <div>
            <div className="card">
              <div className="card-header">
                <h3><ImageIcon size={16} color="var(--accent-blue)" /> Evidence Input</h3>
                {file && (
                  <button className="btn btn-ghost" style={{fontSize: '0.75rem'}} onClick={() => { setFile(null); setPreview(null); setResults(null); }}>
                    Clear
                  </button>
                )}
              </div>
              <div className="card-body">
                {!preview ? (
                  <div
                    className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
                    onDragEnter={handleDrag} onDragLeave={handleDrag}
                    onDragOver={handleDrag} onDrop={handleDrop}
                    onClick={() => fileInputRef.current.click()}
                  >
                    <div className="icon"><UploadCloud size={28} /></div>
                    <h3>Drop traffic image here</h3>
                    <p>Supports JPG, PNG, WebP • Up to 10MB</p>
                    <input ref={fileInputRef} type="file" accept="image/*"
                      style={{display:'none'}} onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
                  </div>
                ) : (
                  <div style={{position:'relative'}}>
                    <div className="image-container">
                      <img src={preview} alt="Evidence" />

                      {/* Render detector-grounded violation boxes */}
                      {results?.violations?.map((v, i) => {
                        const bbox = v.bbox_percent || {};
                        return (
                          <div key={i} className="bounding-box" style={{
                            left: `${bbox.x || 0}%`,
                            top: `${bbox.y || 0}%`,
                            width: `${bbox.w || 0}%`,
                            height: `${bbox.h || 0}%`
                          }}>
                            <div className="bounding-box-label">
                              {v.type} ({(v.confidence*100).toFixed(0)}%)
                            </div>
                          </div>
                        );
                      })}

                      {/* Loading overlay */}
                      {isAnalyzing && (
                        <div className="analysis-overlay">
                          <div className="loader loader-lg"></div>
                          <div style={{fontWeight:600, fontSize:'0.95rem'}}>{analysisStep}</div>
                          <div style={{color:'var(--text-muted)', fontSize:'0.75rem'}}>
                            Detector + OCR + rule engine
                          </div>
                        </div>
                      )}
                    </div>

                    <div style={{display:'flex', gap:'0.75rem', marginTop:'1rem'}}>
                      <button className="btn btn-primary" style={{flex:1}}
                        onClick={analyzeImage} disabled={isAnalyzing}>
                        {isAnalyzing ? <><div className="loader"></div> Analyzing...</> : <><Zap size={16} /> Run Analysis</>}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Preprocessing Info */}
            {results?.preprocessing && (
              <div className="card" style={{marginTop:'1rem'}}>
                <div className="card-header">
                  <h3><Zap size={16} color="var(--accent-cyan)" /> Image Preprocessing</h3>
                </div>
                <div className="card-body">
                  <div className="preprocessing-tags">
                    {results.preprocessing.steps_applied.map((step, i) => (
                      <span key={i} className="preprocessing-tag">{step}</span>
                    ))}
                  </div>
                  {results.preprocessing.quality_metrics && (
                    <div className="scene-info" style={{marginTop:'1rem'}}>
                      <div className="scene-item">
                        <div className="label">Brightness</div>
                        <div className="value">{results.preprocessing.quality_metrics.brightness}</div>
                      </div>
                      <div className="scene-item">
                        <div className="label">Contrast</div>
                        <div className="value">{results.preprocessing.quality_metrics.contrast}</div>
                      </div>
                      <div className="scene-item">
                        <div className="label">Sharpness</div>
                        <div className="value">{results.preprocessing.quality_metrics.sharpness}</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {results?.pipeline && (
              <div className="card" style={{marginTop:'1rem'}}>
                <div className="card-header">
                  <h3><Search size={16} color="var(--accent-emerald)" /> Pipeline</h3>
                </div>
                <div className="card-body">
                  <div className="scene-info">
                    <div className="scene-item">
                      <div className="label">Detector</div>
                      <div className="value">{results.pipeline.detector?.model || results.pipeline.detector?.backend || 'unavailable'}</div>
                    </div>
                    <div className="scene-item">
                      <div className="label">OCR Plates</div>
                      <div className="value">{results.detection?.recognized_plates?.length || 0}</div>
                    </div>
                    <div className="scene-item">
                      <div className="label">Vision Review</div>
                      <div className="value">{results.pipeline.vision_review?.configured ? results.pipeline.vision_review.provider : 'off'}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div>
            {/* Scene Info */}
            {results?.scene && (
              <div className="card" style={{marginBottom:'1rem'}}>
                <div className="card-header">
                  <h3><Eye size={16} color="var(--accent-purple)" /> Scene Analysis</h3>
                </div>
                <div className="card-body">
                  <p style={{fontSize:'0.85rem', color:'var(--text-secondary)', marginBottom:'1rem'}}>
                    {results.scene.description}
                  </p>
                  <div className="scene-info">
                    <div className="scene-item">
                      <div className="label">Weather</div>
                      <div className="value">{results.scene.weather}</div>
                    </div>
                    <div className="scene-item">
                      <div className="label">Traffic Light</div>
                      <div className="value" style={{color: results.scene.traffic_light === 'red' ? 'var(--accent-rose)' : results.scene.traffic_light === 'green' ? 'var(--accent-emerald)' : 'var(--text-secondary)'}}>
                        {results.scene.traffic_light}
                      </div>
                    </div>
                    <div className="scene-item">
                      <div className="label">Vehicles</div>
                      <div className="value">{results.detection?.total_vehicles}</div>
                    </div>
                    <div className="scene-item">
                      <div className="label">Pedestrians</div>
                      <div className="value">{results.detection?.total_pedestrians}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Violation Report */}
            <div className="card">
              <div className="card-header">
                <h3><AlertTriangle size={16} color="var(--accent-rose)" /> Violation Report</h3>
                {results && (
                  <span className="badge badge-danger">{results.violations?.length || 0} found</span>
                )}
              </div>
              <div className="card-body">
                {!results && !isAnalyzing && (
                  <div className="empty-state">
                    <div className="icon">🔍</div>
                    <p>Upload an image and run analysis to detect violations.</p>
                  </div>
                )}

                {isAnalyzing && (
                  <div className="empty-state">
                    <div className="loader loader-lg" style={{margin:'0 auto 1rem'}}></div>
                    <p>Running detector, OCR, and violation rules...</p>
                  </div>
                )}

                {results?.violations?.length === 0 && (
                  <div style={{padding:'1.5rem', background:'rgba(16,185,129,0.08)', border:'1px solid rgba(16,185,129,0.2)', borderRadius:'8px', textAlign:'center'}}>
                    <CheckCircle size={24} color="var(--accent-emerald)" style={{marginBottom:'8px'}} />
                    <p style={{color:'var(--accent-emerald)', fontWeight:600}}>No violations detected</p>
                    <p style={{color:'var(--text-muted)', fontSize:'0.8rem', marginTop:'4px'}}>All road users appear compliant in this frame.</p>
                  </div>
                )}

                {results?.violations?.map((v, i) => (
                  <div key={i} style={{
                    padding:'1rem', marginBottom:'0.75rem',
                    background:'rgba(244,63,94,0.04)', border:'1px solid rgba(244,63,94,0.1)',
                    borderRadius:'10px'
                  }}>
                    <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'8px'}}>
                      <span className="badge badge-danger">{v.type}</span>
                      <span style={{fontSize:'0.75rem', color:'var(--text-muted)'}}>
                        {v.vehicle_type}
                      </span>
                    </div>

                    {v.license_plate && v.license_plate !== 'null' && (
                      <div style={{marginBottom:'8px'}}>
                        <span style={{fontSize:'0.7rem', color:'var(--text-muted)'}}>LICENSE PLATE: </span>
                        <span className="plate-text">{v.license_plate}</span>
                      </div>
                    )}

                    <div style={{marginBottom:'6px'}}>
                      <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.75rem', marginBottom:'3px'}}>
                        <span style={{color:'var(--text-muted)'}}>Confidence</span>
                        <span style={{fontFamily:'JetBrains Mono', fontWeight:600}}>{(v.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="confidence-bar">
                        <div className={`confidence-fill ${getConfClass(v.confidence)}`}
                          style={{width:`${v.confidence*100}%`}}></div>
                      </div>
                    </div>

                    {v.description && (
                      <div className="violation-desc">"{v.description}"</div>
                    )}
                  </div>
                ))}

                {/* Evidence link */}
                {results?.evidence && (
                  <div style={{marginTop:'1rem', padding:'0.75rem', background:'rgba(59,130,246,0.06)', border:'1px solid rgba(59,130,246,0.15)', borderRadius:'8px'}}>
                    <div style={{display:'flex', alignItems:'center', gap:'8px', fontSize:'0.8rem'}}>
                      <FileText size={14} color="var(--accent-blue)" />
                      <a href={`${API}${results.evidence.annotated_image}`} target="_blank" rel="noopener noreferrer" style={{color:'var(--accent-blue)', textDecoration:'none', fontWeight:600}}>
                        Open annotated evidence image
                      </a>
                    </div>
                    <div style={{fontSize:'0.75rem', color:'var(--text-muted)', marginTop:'4px'}}>
                      Session: {results.session_id} • Processing: {(results.processing_time_ms/1000).toFixed(1)}s
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
