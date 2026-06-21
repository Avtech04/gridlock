import './index.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const DASHBOARD_URL = `${API_BASE}/ui/?v=command-center-20260621-9`;

function App() {
  return (
    <main className="react-bridge">
      <header className="bridge-header">
        <div>
          <span>Gridlock AI React Shell</span>
          <h1>Opening the production dashboard</h1>
          <p>
            The FastAPI-served dashboard is the single source of truth for the final UI, OCR,
            review workflow, analytics brief, ZIP batch mode, and performance tab.
          </p>
        </div>
        <a href={DASHBOARD_URL} target="_blank" rel="noreferrer">
          Open Full Dashboard
        </a>
      </header>
      <iframe
        title="Gridlock AI production dashboard"
        src={DASHBOARD_URL}
        className="bridge-frame"
      />
    </main>
  );
}

export default App;
