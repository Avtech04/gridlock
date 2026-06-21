# Gridlock AI Frontend

This folder contains the optional React development shell for Gridlock AI.
The main production dashboard is served by the FastAPI backend from `web/` at:

```text
http://localhost:8000/ui/
```

Use the React shell only when actively developing frontend components. For
hackathon review and normal testing, run the backend and open `/ui/`.

## Development

```bash
cd frontend
npm install
npm run dev
```

The dashboard communicates with the backend APIs, so keep the FastAPI server
running on port `8000`.

## Build

```bash
cd frontend
npm run build
```

Generated build artifacts are intentionally excluded from the source submission.
