# Run Gridlock AI On A New System

Use this guide when the source code and model weights are transferred to another machine.

## Required Files

1. Source code:

```text
gridlock_source_submission.zip
```

2. Model weights:

```text
gridlock_model_weights_with_paths.zip
```

The model zip may be uploaded separately or shared through Google Drive. Download it before starting the setup.

## Expected Model Paths

After extracting the model zip from the project root, these files should exist:

```text
backend/models/traffic_yolo26s_best.pt
backend/models/helmet_violations_best.pt
backend/yolo26s.pt
backend/yolov3-tiny.weights
```

`traffic_yolo26s_best.pt` is the primary trained traffic-violation model. `helmet_violations_best.pt` is an additional specialist safety model. `yolo26s.pt` and `yolov3-tiny.weights` are fallback models.

## macOS / Linux Setup

```bash
unzip gridlock_source_submission.zip -d gridlock-hackathon
cd gridlock-hackathon
unzip ../gridlock_model_weights_with_paths.zip -d .
./start.sh
```

Open:

```text
http://localhost:8000/ui/
```

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Windows PowerShell Setup

```powershell
Expand-Archive gridlock_source_submission.zip -DestinationPath gridlock-hackathon
cd gridlock-hackathon
Expand-Archive ..\gridlock_model_weights_with_paths.zip -DestinationPath .
.\start.bat
```

Open:

```text
http://localhost:8000/ui/
```

Manual equivalent:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## OCR Setup

Plate OCR uses Tesseract by default.

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
2. Add the install folder, usually `C:\Program Files\Tesseract-OCR`, to PATH.
3. Restart PowerShell.

Optional OCR engines can be installed with:

```bash
pip install -r backend/requirements-optional-ocr.txt
```

## Demo Test

Single-image test folders:

```text
testing_usecases/judge_ready/
```

Batch upload test:

```text
testing_usecases/gridlock_batch_upload_demo.zip
```

Recommended sequence:

1. Open the Analyze tab.
2. Upload one image from `testing_usecases/judge_ready/helmet_non_compliance/`.
3. Upload one image from `testing_usecases/judge_ready/license_plate_ocr/`.
4. Switch to ZIP batch mode.
5. Upload `testing_usecases/gridlock_batch_upload_demo.zip`.
6. Open Records, Analytics, and Performance.

## Optional Gemini / Groq Brief

The app works without an API key. To enable AI-generated analytics summaries:

1. Open Config.
2. Paste a Gemini or Groq API key.
3. Save the key.
4. Refresh Analytics.

Without a key, Analytics uses a local deterministic summary.

## Troubleshooting

If the detector status does not show trained models:

1. Confirm the model zip was extracted from the project root.
2. Confirm these paths exist:

```text
backend/models/traffic_yolo26s_best.pt
backend/models/helmet_violations_best.pt
```

3. Restart the backend.

If OCR does not read plates, confirm the Tesseract binary is installed and available on PATH.
