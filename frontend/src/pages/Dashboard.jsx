import { useState, useEffect } from 'react';
import axios from 'axios';
import { AlertTriangle, Car, Clock } from 'lucide-react';

const API = 'http://localhost:8000';

export default function DashboardPage() {
  const [analytics, setAnalytics] = useState(null);
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    axios.get(`${API}/api/analytics`).then(r => setAnalytics(r.data)).catch(() => {});
    axios.get(`${API}/api/sessions?limit=5`).then(r => setSessions(r.data.sessions)).catch(() => {});
  }, []);

  return (
    <>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Real-time overview of traffic violation detection system</p>
      </div>

      <div className="page-content">
        <div className="stats-grid">
          <div className="stat-card blue">
            <div className="stat-label">Images Analyzed</div>
            <div className="stat-value">{analytics?.summary?.total_sessions || 0}</div>
            <div className="stat-detail">Total sessions processed</div>
          </div>
          <div className="stat-card rose">
            <div className="stat-label">Violations Detected</div>
            <div className="stat-value">{analytics?.summary?.total_violations || 0}</div>
            <div className="stat-detail">Across all analyses</div>
          </div>
          <div className="stat-card emerald">
            <div className="stat-label">Avg Processing Time</div>
            <div className="stat-value">{analytics?.performance?.avg_processing_ms ? `${(analytics.performance.avg_processing_ms / 1000).toFixed(1)}s` : '—'}</div>
            <div className="stat-detail">Per image analysis</div>
          </div>
          <div className="stat-card amber">
            <div className="stat-label">Top Violation</div>
            <div className="stat-value" style={{fontSize: '1.1rem'}}>
              {analytics?.violations_by_type?.[0]?.type || '—'}
            </div>
            <div className="stat-detail">
              {analytics?.violations_by_type?.[0] ? `${analytics.violations_by_type[0].count} occurrences` : 'No data'}
            </div>
          </div>
        </div>

        <div className="grid-2">
          {/* Violations by Type */}
          <div className="card">
            <div className="card-header">
              <h3><AlertTriangle size={16} color="var(--accent-rose)" /> Violations by Type</h3>
            </div>
            <div className="card-body">
              {analytics?.violations_by_type?.length > 0 ? (
                analytics.violations_by_type.map((item, i) => (
                  <div key={i} style={{marginBottom: '0.75rem'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px'}}>
                      <span>{item.type}</span>
                      <span style={{fontWeight: 600, fontFamily: 'JetBrains Mono'}}>{item.count}</span>
                    </div>
                    <div className="confidence-bar">
                      <div className="confidence-fill high" style={{
                        width: `${(item.count / Math.max(...analytics.violations_by_type.map(v=>v.count))) * 100}%`
                      }}></div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-state">
                  <div className="icon">📊</div>
                  <p>No violation data yet. Analyze images to see stats.</p>
                </div>
              )}
            </div>
          </div>

          {/* Top Offending Plates */}
          <div className="card">
            <div className="card-header">
              <h3><Car size={16} color="var(--accent-amber)" /> Top Offending Vehicles</h3>
            </div>
            <div className="card-body">
              {analytics?.top_offending_plates?.length > 0 ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>License Plate</th>
                      <th>Violations</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.top_offending_plates.map((p, i) => (
                      <tr key={i}>
                        <td><span className="plate-text">{p.plate}</span></td>
                        <td><span className="badge badge-danger">{p.count}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="empty-state">
                  <div className="icon">🚗</div>
                  <p>No plate data yet.</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Recent Sessions */}
        <div className="card" style={{marginTop: '1.5rem'}}>
          <div className="card-header">
            <h3><Clock size={16} color="var(--accent-cyan)" /> Recent Analysis Sessions</h3>
          </div>
          <div className="card-body">
            {sessions.length > 0 ? (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Session ID</th>
                    <th>Vehicles</th>
                    <th>Violations</th>
                    <th>Processing</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.map(s => (
                    <tr key={s.id}>
                      <td style={{fontFamily: 'JetBrains Mono', fontSize: '0.8rem'}}>{s.id}</td>
                      <td><span className="badge badge-info">{s.total_vehicles}</span></td>
                      <td><span className="badge badge-danger">{s.total_violations}</span></td>
                      <td>{(s.processing_time_ms / 1000).toFixed(1)}s</td>
                      <td style={{color: 'var(--text-muted)', fontSize: '0.8rem'}}>
                        {new Date(s.timestamp).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty-state">
                <div className="icon">📷</div>
                <p>No sessions yet. Go to "Analyze Image" to get started.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
