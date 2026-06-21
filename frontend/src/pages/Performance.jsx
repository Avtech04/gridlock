import { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Target, Gauge, Timer, BarChart3 } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

const API = 'http://localhost:8000';

function MetricGauge({ label, value, color, suffix = '%' }) {
  const pct = Math.min(value * 100, 100);
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ position: 'relative', width: 100, height: 100, margin: '0 auto' }}>
        <svg viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
          <circle cx="18" cy="18" r="15.5" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
          <circle cx="18" cy="18" r="15.5" fill="none" stroke={color} strokeWidth="3"
            strokeDasharray={`${pct} 100`} strokeLinecap="round" />
        </svg>
        <div style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: '1rem', color
        }}>
          {(value * 100).toFixed(1)}{suffix}
        </div>
      </div>
      <div style={{ marginTop: 8, fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>{label}</div>
    </div>
  );
}

export default function PerformancePage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API}/api/performance`).then(r => {
      if (!r.data.error) setData(r.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <>
      <div className="page-header">
        <h2>Performance Evaluation</h2>
        <p>Model accuracy metrics, throughput, and per-class analysis</p>
      </div>
      <div className="page-content">
        <div className="empty-state"><div className="loader loader-lg" style={{ margin: '0 auto' }}></div></div>
      </div>
    </>
  );

  if (!data) return (
    <>
      <div className="page-header">
        <h2>Performance Evaluation</h2>
        <p>Model accuracy metrics, throughput, and per-class analysis</p>
      </div>
      <div className="page-content">
        <div className="empty-state">
          <div className="icon">📊</div>
          <p>No data yet. Analyze some images first to see performance metrics.</p>
        </div>
      </div>
    </>
  );

  if (data.status === 'needs_ground_truth') return (
    <>
      <div className="page-header">
        <h2>Performance Evaluation</h2>
        <p>Ground-truth metrics and computational efficiency</p>
      </div>
      <div className="page-content">
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-header">
            <h3><Target size={16} color="var(--accent-amber)" /> Ground Truth Required</h3>
          </div>
          <div className="card-body">
            <p style={{color:'var(--text-secondary)', fontSize:'0.9rem', lineHeight:1.6}}>
              {data.message}
            </p>
            <table className="data-table" style={{marginTop:'1rem'}}>
              <tbody>
                <tr><td style={{fontWeight:600}}>Detector</td><td>{data.detector?.model || data.detector?.backend || 'unavailable'}</td></tr>
                <tr><td style={{fontWeight:600}}>Images Processed</td><td>{data.throughput.total_images_processed}</td></tr>
                <tr><td style={{fontWeight:600}}>Avg Processing Time</td><td>{(data.throughput.avg_processing_ms / 1000).toFixed(2)}s</td></tr>
                <tr><td style={{fontWeight:600}}>Throughput</td><td>{data.throughput.images_per_minute} images/min</td></tr>
              </tbody>
            </table>
            <div style={{marginTop:'1rem', padding:'0.75rem', background:'rgba(59,130,246,0.06)', border:'1px solid rgba(59,130,246,0.15)', borderRadius:'8px', color:'var(--text-secondary)', fontSize:'0.8rem'}}>
              Run: <span style={{fontFamily:'JetBrains Mono'}}>python backend/scripts/evaluate_dataset.py --images path/to/images --ground-truth path/to/ground_truth.json --write-db</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );

  const radarData = [
    { metric: 'Accuracy', value: data.overall.accuracy * 100 },
    { metric: 'Precision', value: data.overall.precision * 100 },
    { metric: 'Recall', value: data.overall.recall * 100 },
    { metric: 'F1-Score', value: data.overall.f1_score * 100 },
    { metric: 'mAP50', value: (data.overall.mAP50 ?? data.overall.mAP ?? 0) * 100 },
  ];

  return (
    <>
      <div className="page-header">
        <h2>Performance Evaluation</h2>
        <p>Accuracy, Precision, Recall, F1-score, mAP50, and computational efficiency</p>
      </div>

      <div className="page-content">
        {/* Overall Metrics Gauges */}
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-header">
            <h3><Target size={16} color="var(--accent-blue)" /> Model Performance Metrics</h3>
          </div>
          <div className="card-body">
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: '1.5rem' }}>
              <MetricGauge label="Accuracy" value={data.overall.accuracy} color="#3b82f6" />
              <MetricGauge label="Precision" value={data.overall.precision} color="#10b981" />
              <MetricGauge label="Recall" value={data.overall.recall} color="#f59e0b" />
              <MetricGauge label="F1-Score" value={data.overall.f1_score} color="#8b5cf6" />
              <MetricGauge label="mAP50" value={data.overall.mAP50 ?? data.overall.mAP ?? 0} color="#06b6d4" />
            </div>
          </div>
        </div>

        <div className="grid-2">
          {/* Radar Chart */}
          <div className="card">
            <div className="card-header">
              <h3><Activity size={16} color="var(--accent-purple)" /> Performance Radar</h3>
            </div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Radar name="Score" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.25} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="card">
            <div className="card-header">
              <h3><Gauge size={16} color="var(--accent-emerald)" /> Confusion Matrix</h3>
            </div>
            <div className="card-body">
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem',
                maxWidth: '300px', margin: '0 auto'
              }}>
                <div style={{ padding: '1rem', background: 'rgba(16,185,129,0.12)', borderRadius: 10, textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--accent-emerald)' }}>
                    {data.confusion_matrix.true_positives}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>True Positives</div>
                </div>
                <div style={{ padding: '1rem', background: 'rgba(244,63,94,0.12)', borderRadius: 10, textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--accent-rose)' }}>
                    {data.confusion_matrix.false_positives}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>False Positives</div>
                </div>
                <div style={{ padding: '1rem', background: 'rgba(245,158,11,0.12)', borderRadius: 10, textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--accent-amber)' }}>
                    {data.confusion_matrix.false_negatives}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>False Negatives (est.)</div>
                </div>
                <div style={{ padding: '1rem', background: 'rgba(59,130,246,0.12)', borderRadius: 10, textAlign: 'center' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--accent-blue)' }}>
                    {data.confusion_matrix.true_negatives}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>True Negatives</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid-2" style={{ marginTop: '1.5rem' }}>
          {/* Throughput */}
          <div className="card">
            <div className="card-header">
              <h3><Timer size={16} color="var(--accent-amber)" /> Computational Efficiency</h3>
            </div>
            <div className="card-body">
              <table className="data-table">
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Avg Processing Time</td>
                    <td style={{ fontFamily: 'JetBrains Mono' }}>{(data.throughput.avg_processing_ms / 1000).toFixed(2)}s</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Min Processing Time</td>
                    <td style={{ fontFamily: 'JetBrains Mono' }}>{(data.throughput.min_processing_ms / 1000).toFixed(2)}s</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Max Processing Time</td>
                    <td style={{ fontFamily: 'JetBrains Mono' }}>{(data.throughput.max_processing_ms / 1000).toFixed(2)}s</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Throughput</td>
                    <td style={{ fontFamily: 'JetBrains Mono', color: 'var(--accent-emerald)' }}>
                      {data.throughput.images_per_minute} images/min
                    </td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Total Processed</td>
                    <td style={{ fontFamily: 'JetBrains Mono' }}>{data.throughput.total_images_processed} images</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-class Metrics */}
          <div className="card">
            <div className="card-header">
              <h3><BarChart3 size={16} color="var(--accent-cyan)" /> Per-Class Detection Metrics</h3>
            </div>
            <div className="card-body">
              {data.per_class_metrics.length > 0 ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Violation Type</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>AP50</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.per_class_metrics.map((pc, i) => (
                      <tr key={i}>
                        <td style={{ fontSize: '0.8rem' }}>{pc.type}</td>
                        <td style={{ fontFamily: 'JetBrains Mono', color: pc.precision >= 0.8 ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}>
                          {(pc.precision * 100).toFixed(0)}%
                        </td>
                        <td style={{ fontFamily: 'JetBrains Mono' }}>{(pc.recall * 100).toFixed(0)}%</td>
                        <td style={{ fontFamily: 'JetBrains Mono' }}>{(pc.f1_score * 100).toFixed(0)}%</td>
                        <td style={{ fontFamily: 'JetBrains Mono' }}>{(pc.ap50 * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="empty-state"><p>No per-class data yet.</p></div>
              )}
            </div>
          </div>
        </div>

        {/* Confidence Distribution */}
        {data.confidence_distribution && (
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <div className="card-header">
            <h3><BarChart3 size={16} color="var(--accent-rose)" /> Confidence Threshold Analysis</h3>
          </div>
          <div className="card-body">
            <div style={{ display: 'flex', gap: '1rem' }}>
              <div style={{ flex: 1, padding: '1rem', background: 'rgba(16,185,129,0.08)', borderRadius: 10, textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--accent-emerald)' }}>
                  {data.confidence_distribution.high_confidence_gte_80}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>High (&ge;80%)</div>
              </div>
              <div style={{ flex: 1, padding: '1rem', background: 'rgba(245,158,11,0.08)', borderRadius: 10, textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--accent-amber)' }}>
                  {data.confidence_distribution.medium_confidence_50_80}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Medium (50-80%)</div>
              </div>
              <div style={{ flex: 1, padding: '1rem', background: 'rgba(244,63,94,0.08)', borderRadius: 10, textAlign: 'center' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--accent-rose)' }}>
                  {data.confidence_distribution.low_confidence_lt_50}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Low (&lt;50%)</div>
              </div>
            </div>
            {data.confidence_thresholds?.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={data.confidence_thresholds} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="threshold" tick={{ fill: '#94a3b8', fontSize: 11 }}
                      tickFormatter={v => `${v * 100}%`} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9' }} />
                    <Bar dataKey="detections" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Detections above threshold" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
        )}
      </div>
    </>
  );
}
