import { useState, useEffect } from 'react';
import axios from 'axios';
import { AlertTriangle, Search, ChevronLeft, ChevronRight, Trash2 } from 'lucide-react';

const API = 'http://localhost:8000';

export default function ViolationsPage() {
  const [violations, setViolations] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('');

  const violationTypes = [
    'Helmet Non-compliance', 'Seatbelt Non-compliance', 'Triple Riding',
    'Wrong-side Driving', 'Stop-line Violation', 'Red-light Violation', 'Illegal Parking'
  ];

  const fetchViolations = () => {
    const params = { page, limit: 15 };
    if (search) params.search = search;
    if (filterType) params.violation_type = filterType;
    axios.get(`${API}/api/violations`, { params }).then(r => {
      setViolations(r.data.violations);
      setTotal(r.data.total);
      setTotalPages(r.data.total_pages);
    }).catch(() => {});
  };

  // Fetch only on pagination/filter changes. Text search is submitted explicitly.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { fetchViolations(); }, [page, filterType]);

  const handleSearch = (e) => {
    e.preventDefault();
    setPage(1);
    fetchViolations();
  };

  const clearAll = async () => {
    if (!confirm('Delete all violation records?')) return;
    await axios.delete(`${API}/api/violations`);
    fetchViolations();
  };

  const getConfClass = (c) => c >= 0.8 ? 'high' : c >= 0.5 ? 'medium' : 'low';

  return (
    <>
      <div className="page-header">
        <h2>Violation Records</h2>
        <p>Searchable database of all detected traffic violations</p>
      </div>

      <div className="page-content">
        <div className="card">
          <div className="card-header">
            <h3><AlertTriangle size={16} color="var(--accent-rose)" /> All Violations ({total})</h3>
            <div style={{display:'flex', gap:'0.5rem', alignItems:'center'}}>
              <form onSubmit={handleSearch} className="search-wrapper">
                <Search size={14} />
                <input className="search-input" placeholder="Search plate or description..."
                  value={search} onChange={e => setSearch(e.target.value)} />
              </form>
              <select value={filterType} onChange={e => { setFilterType(e.target.value); setPage(1); }}
                style={{
                  background:'rgba(255,255,255,0.05)', border:'1px solid var(--border)',
                  borderRadius:'8px', padding:'8px 12px', color:'var(--text-primary)',
                  fontSize:'0.8rem', cursor:'pointer'
                }}>
                <option value="">All Types</option>
                {violationTypes.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              <button className="btn btn-danger" onClick={clearAll} style={{padding:'8px'}}>
                <Trash2 size={14} />
              </button>
            </div>
          </div>
          <div className="card-body" style={{padding:0}}>
            {violations.length > 0 ? (
              <>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Violation Type</th>
                      <th>License Plate</th>
                      <th>Vehicle</th>
                      <th>Confidence</th>
                      <th>Description</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {violations.map(v => (
                      <tr key={v.id}>
                        <td style={{fontFamily:'JetBrains Mono', fontSize:'0.75rem', color:'var(--text-muted)'}}>{v.id}</td>
                        <td><span className="badge badge-danger">{v.violation_type}</span></td>
                        <td>
                          {v.license_plate && v.license_plate !== 'null' ? (
                            <span className="plate-text">{v.license_plate}</span>
                          ) : <span style={{color:'var(--text-muted)'}}>—</span>}
                        </td>
                        <td style={{fontSize:'0.85rem'}}>{v.vehicle_type || '—'}</td>
                        <td>
                          <div style={{display:'flex', alignItems:'center', gap:'6px'}}>
                            <span style={{fontFamily:'JetBrains Mono', fontSize:'0.8rem', fontWeight:600}}>
                              {(v.confidence*100).toFixed(0)}%
                            </span>
                            <div className="confidence-bar" style={{width:'60px'}}>
                              <div className={`confidence-fill ${getConfClass(v.confidence)}`}
                                style={{width:`${v.confidence*100}%`}}></div>
                            </div>
                          </div>
                        </td>
                        <td style={{maxWidth:'200px', fontSize:'0.75rem', color:'var(--text-secondary)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>
                          {v.description || '—'}
                        </td>
                        <td style={{fontSize:'0.75rem', color:'var(--text-muted)', whiteSpace:'nowrap'}}>
                          {new Date(v.timestamp).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* Pagination */}
                <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', padding:'1rem'}}>
                  <span style={{fontSize:'0.8rem', color:'var(--text-muted)'}}>
                    Showing {(page-1)*15 + 1}–{Math.min(page*15, total)} of {total}
                  </span>
                  <div style={{display:'flex', gap:'0.5rem'}}>
                    <button className="btn btn-ghost" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
                      <ChevronLeft size={14} /> Prev
                    </button>
                    <button className="btn btn-ghost" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>
                      Next <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="icon">📋</div>
                <p>No violations found. {search || filterType ? 'Try adjusting your filters.' : 'Analyze images to populate this database.'}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
