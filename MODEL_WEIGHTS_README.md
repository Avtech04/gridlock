# Gridlock AI Model Weights Package

This package contains the trained detector weights and fallback model assets used
by Gridlock AI.

Extract this zip from the project root after extracting the source package:

```bash
unzip gridlock_model_weights_with_paths.zip -d .
```

Windows PowerShell:

```powershell
Expand-Archive gridlock_model_weights_with_paths.zip -DestinationPath .
```

After extraction, these files should exist:

```text
backend/models/traffic_yolo26s_best.pt
backend/models/helmet_violations_best.pt
backend/yolo26s.pt
backend/yolov3-tiny.weights
```

The backend reads `backend/config.yaml` and automatically loads the strongest
available detector candidates. Start the application only after this package has
been extracted.
