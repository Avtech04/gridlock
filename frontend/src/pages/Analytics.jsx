import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart3, TrendingUp } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';

const API = 'http://localhost:8000';

const COLORS = ['#f43f5e', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899'];

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState(null);

  useEffect(() => {
    axios.get(`${API}/api/analytics`).then(r => setAnalytics(r.data)).catch(() => {});
  }, []);

  if (!analytics) return (
    <>
      <div className="page-header">
        <h2>Analytics & Reporting</h2>
        <p>Comprehensive violation statistics and trends</p>
      </div>
      <div className="page-content">
        <div className="empty-state">
          <div className="loader loader-lg" style={{margin:'0 auto'}}></div>
          <p style={{marginTop:'1rem'}}>Loading analytics...</p>
        </div>
      </div>
    </>
  );

  const barData = analytics.violations_by_type?.map(v => ({
    name: v.type.replace('Non-compliance', 'NC').replace('Violation', 'Viol.'),
    count: v.count,
    fullName: v.type
  })) || [];

  const pieData = analytics.violations_by_vehicle?.map(v => ({
    name: v.vehicle || 'Unknown',
    value: v.count
  })) || [];

  const confData = analytics.avg_confidence_by_type?.map(v => ({
    name: v.type.replace('Non-compliance', 'NC').replace('Violation', 'Viol.'),
    confidence: Math.round(v.confidence * 100)
  })) || [];

  return (
    <>
      <div className="page-header">
        <h2>Analytics & Reporting</h2>
        <p>Comprehensive violation statistics, trends, and performance metrics</p>
      </div>

      <div className="page-content">
        {/* Summary Stats */}
        <div className="stats-grid">
          <div className="stat-card blue">
            <div className="stat-label">Total Analyses</div>
            <div className="stat-value">{analytics.summary?.total_sessions || 0}</div>
          </div>
          <div className="stat-card rose">
            <div className="stat-label">Total Violations</div>
            <div className="stat-value">{analytics.summary?.total_violations || 0}</div>
          </div>
          <div className="stat-card emerald">
            <div className="stat-label">Avg Processing</div>
            <div className="stat-value">{analytics.performance?.avg_processing_ms ? `${(analytics.performance.avg_processing_ms / 1000).toFixed(1)}s` : '—'}</div>
          </div>
          <div className="stat-card amber">
            <div className="stat-label">Min / Max Processing</div>
            <div className="stat-value" style={{fontSize:'1rem'}}>
              {analytics.performance?.min_processing_ms ? `${(analytics.performance.min_processing_ms / 1000).toFixed(1)}s / ${(analytics.performance.max_processing_ms / 1000).toFixed(1)}s` : '—'}
            </div>
          </div>
        </div>

        <div className="grid-2">
          {/* Bar Chart: Violations by Type */}
          <div className="card">
            <div className="card-header">
              <h3><BarChart3 size={16} color="var(--accent-blue)" /> Violations by Type</h3>
            </div>
            <div className="card-body">
              {barData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={barData} margin={{top:5, right:10, left:-20, bottom:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" tick={{fill:'#94a3b8', fontSize:11}} />
                    <YAxis tick={{fill:'#94a3b8', fontSize:11}} />
                    <Tooltip
                      contentStyle={{background:'#1e293b', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px', color:'#f1f5f9'}}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName}
                    />
                    <Bar dataKey="count" fill="#3b82f6" radius={[4,4,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state"><p>No data yet.</p></div>
              )}
            </div>
          </div>

          {/* Pie Chart: By Vehicle Type */}
          <div className="card">
            <div className="card-header">
              <h3><TrendingUp size={16} color="var(--accent-purple)" /> Violations by Vehicle Type</h3>
            </div>
            <div className="card-body">
              {pieData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" outerRadius={100} dataKey="value"
                      label={({name, percent}) => `${name} (${(percent*100).toFixed(0)}%)`}
                      labelLine={false}
                    >
                      {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip contentStyle={{background:'#1e293b', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px', color:'#f1f5f9'}} />
                    <Legend wrapperStyle={{fontSize:'0.8rem', color:'#94a3b8'}} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state"><p>No data yet.</p></div>
              )}
            </div>
          </div>
        </div>

        {/* Confidence Chart */}
        <div className="card" style={{marginTop:'1.5rem'}}>
          <div className="card-header">
            <h3><BarChart3 size={16} color="var(--accent-emerald)" /> Average Confidence by Violation Type</h3>
          </div>
          <div className="card-body">
            {confData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={confData} margin={{top:5, right:10, left:-20, bottom:5}}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="name" tick={{fill:'#94a3b8', fontSize:11}} />
                  <YAxis domain={[0,100]} tick={{fill:'#94a3b8', fontSize:11}} unit="%" />
                  <Tooltip contentStyle={{background:'#1e293b', border:'1px solid rgba(255,255,255,0.1)', borderRadius:'8px', color:'#f1f5f9'}} />
                  <Bar dataKey="confidence" fill="#10b981" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state"><p>No data yet.</p></div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
