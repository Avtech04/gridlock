# HackerEarth Submission Checklist

## Title

Gridlock AI Enforcer: Automated Traffic Violation Evidence Intelligence

## Description

Use `docs/PROJECT_PROPOSAL.md` as the base description. Keep the description focused on:

- traffic image preprocessing;
- YOLO-based vehicle, road-user, helmet, seatbelt, plate, signal, stop-line, wrong-side, and parking evidence detection;
- license plate OCR;
- human approval/removal of false detections;
- annotated evidence generation;
- searchable records, analytics, AI command brief, and performance evaluation workflow.

## Snapshots

Upload screenshots from:

- Analyze tab with annotated evidence;
- ZIP batch results;
- Records tab;
- Analytics tab with AI Command Brief;
- Performance tab showing operational readiness and evaluation path.

## Video URL

Upload or link a short demo video showing:

1. Open `http://localhost:8000/ui/`.
2. Upload one curated test image.
3. Show plate OCR and violation cards.
4. Remove one false detection if present.
5. Approve evidence.
6. Upload `testing_usecases/gridlock_batch_upload_demo.zip`.
7. Open Analytics and Performance.

## Presentation

Use `docs/PROJECT_PROPOSAL.md`, `docs/FEATURE_COVERAGE.md`, and `docs/TRAINING_RESULTS.md` as content sources.

## Demo Link

Use the deployed URL if available. For local review, use:

```text
http://localhost:8000/ui/
```

## Repository URL

Push the source repo after `.gitignore` cleanup. Do not commit local weights, datasets, virtualenvs, `node_modules`, DB files, uploads, or annotated outputs.

## Source Code

Upload:

```text
submission/gridlock_source_submission.zip
```

This zip is intentionally under the 50 MB limit and excludes model weights.

## Custom Attachment

If the platform allows a larger custom attachment, include model weights separately:

```text
submission/gridlock_model_weights_with_paths.zip
```

If no custom attachment is allowed, upload the model package to Google Drive and
include the Drive link in the run instructions. Reviewers should download the
model zip and extract it from the project root before starting the app.

## Instructions To Run

Full fresh-machine guide:

```text
docs/RUN_ON_NEW_SYSTEM.md
```

Short version:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/ui/
```

Windows:

```bat
start.bat
```
