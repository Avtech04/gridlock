# Gridlock AI Enforcer

Automated photo identification and classification for traffic violations using computer vision.

Gridlock AI Enforcer analyzes traffic evidence images, detects vehicles and road users, identifies violation evidence, extracts readable number plates, generates annotated proof images, and stores searchable records for review and reporting.

## Key Capabilities

- Image preprocessing for low light, shadows, glare, blur, and noisy camera frames.
- YOLO-based detection for vehicles, riders, pedestrians, safety equipment, plates, traffic lights, signs, and violation evidence.
- Violation classification for:
  - Helmet Non-compliance
  - Seatbelt Non-compliance
  - Triple Riding
  - Wrong-side Driving
  - Stop-line Violation
  - Red-light Violation
  - Illegal Parking
- License plate detection and OCR using Tesseract by default, with PaddleOCR/EasyOCR support if installed.
- Annotated evidence images with boxes, labels, confidence scores, plates, and timestamps.
- Human review workflow to remove incorrect detections before approval.
- ZIP batch upload for multi-image processing.
- Records, analytics, AI command brief, and performance evaluation workflow.

## System Architecture

```text
Traffic image
  -> preprocessing
  -> YOLO detector ensemble
  -> plate OCR
  -> violation rule engine
  -> optional Gemini/Groq review
  -> annotated evidence
  -> SQLite records + analytics + performance metrics
```

The runtime is evidence-first: every reported violation is tied to a detected object, confidence score, evidence source, and image region. Optional AI review is used only as an assistive layer; the core workflow works fully without an external API key.

## Project Structure

```text
backend/                  FastAPI backend, detector, OCR, rules, evaluator
web/                      Main dashboard served at /ui/
frontend/                 Optional React shell that points to the same dashboard
docs/                     Proposal, feature coverage, setup, and submission notes
testing_usecases/         Verified demo images and ZIP batch upload package
submission/               Source zip and separate model-weights package
start.sh                  macOS/Linux launcher
start.bat                 Windows launcher
```

## Model Weights

The source-code zip does not include large `.pt` model files. They are supplied separately because the submission source upload has a size limit.

Required model package:

```text
submission/gridlock_model_weights_with_paths.zip
```

If the weights are shared through Google Drive, download that zip and extract it from the project root:

```bash
unzip gridlock_model_weights_with_paths.zip -d .
```

After extraction, these files should exist:

```text
backend/models/traffic_yolo26s_best.pt
backend/models/helmet_violations_best.pt
backend/yolo26s.pt
backend/yolov3-tiny.weights
```

The backend automatically loads the best available model candidates in the order defined in `backend/config.yaml`.

## Quick Start

### macOS / Linux

```bash
unzip gridlock_source_submission.zip -d gridlock-hackathon
cd gridlock-hackathon
unzip gridlock_model_weights_with_paths.zip -d .
./start.sh
```

Open:

```text
http://localhost:8000/ui/
```

### Windows PowerShell

```powershell
Expand-Archive gridlock_source_submission.zip -DestinationPath gridlock-hackathon
cd gridlock-hackathon
Expand-Archive gridlock_model_weights_with_paths.zip -DestinationPath .
.\start.bat
```

Open:

```text
http://localhost:8000/ui/
```

## Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## OCR Dependency

The Python package `pytesseract` is included in `backend/requirements.txt`, but the Tesseract OCR binary must also be installed on the system.

macOS:

```bash
brew install tesseract
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Windows:

1. Install Tesseract OCR from the UB Mannheim Windows build.
2. Add `C:\Program Files\Tesseract-OCR` to PATH.
3. Restart PowerShell.

The app still runs without Tesseract, but plate text extraction is best with it installed.

## Demo Test Files

Verified single-image folders:

```text
testing_usecases/judge_ready/helmet_non_compliance/
testing_usecases/judge_ready/triple_riding/
testing_usecases/judge_ready/license_plate_ocr/
testing_usecases/judge_ready/seatbelt_non_compliance/
testing_usecases/judge_ready/illegal_parking/
```

Verified ZIP batch upload:

```text
testing_usecases/gridlock_batch_upload_demo.zip
```

This ZIP contains 13 verified images across helmet, triple riding, OCR, seatbelt, and illegal parking cases. Use the UI's `ZIP batch` mode to upload it.

## Suggested Demo Flow

1. Start the app and open `http://localhost:8000/ui/`.
2. Upload one helmet or triple-riding image from `testing_usecases/judge_ready/`.
3. Show the annotated evidence and confidence-scored violation cards.
4. Upload `testing_usecases/gridlock_batch_upload_demo.zip` in ZIP batch mode.
5. Open Records to show stored evidence images.
6. Open Analytics to show violation trends and operational brief.
7. Open Performance to show readiness, throughput, and the evaluation workflow.

## Evaluation

Formal metrics are computed from labelled ground truth using:

```bash
python backend/scripts/evaluate_dataset.py \
  --images path/to/eval/images \
  --ground-truth path/to/ground_truth.json \
  --dataset-name gridlock-val \
  --write-db
```

The evaluator reports Accuracy, Precision, Recall, F1-score, and mAP50, and the dashboard reads the saved metrics from SQLite.

Ground-truth JSON format:

```json
{
  "images": [
    {
      "file_name": "traffic_001.jpg",
      "violations": [
        {
          "type": "Helmet Non-compliance",
          "bbox_percent": {"x": 12.5, "y": 34.0, "w": 10.2, "h": 18.7}
        }
      ]
    }
  ]
}
```

## Optional Gemini / Groq Brief

The system works without any API key. To enable AI-generated analytics summaries:

1. Open the Config tab.
2. Paste a Gemini or Groq key.
3. Save the key.
4. Reopen Analytics.

Without a key, the dashboard uses a deterministic local brief generator.

## Submission Files

Use these two files for submission/review:

```text
submission/gridlock_source_submission.zip
submission/gridlock_model_weights_with_paths.zip
```

If the platform cannot accept the model zip directly, upload the model zip to Google Drive and include the Drive link in the run instructions. Reviewers should download it and extract it from the project root before starting the app.

## Additional Documentation

- `docs/PROJECT_PROPOSAL.md`: concept note and solution framework
- `docs/FEATURE_COVERAGE.md`: requirement-by-requirement implementation map
- `docs/RUN_ON_NEW_SYSTEM.md`: fresh-machine setup guide
- `docs/SUBMISSION_CHECKLIST.md`: upload checklist for HackerEarth
- `docs/TRAINING_RESULTS.md`: dataset and model-training summary
- `MODEL_WEIGHTS_README.md`: model package extraction guide
