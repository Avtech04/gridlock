# Model Weights

Large model binaries are distributed separately from the source-code zip.

Extract `gridlock_model_weights_with_paths.zip` from the project root so the
following paths are created:

```text
backend/models/traffic_yolo26s_best.pt
backend/models/helmet_violations_best.pt
backend/yolo26s.pt
backend/yolov3-tiny.weights
```

`traffic_yolo26s_best.pt` is the primary traffic-evidence detector.
`helmet_violations_best.pt` is the specialist rider-safety detector.
The backend reads `backend/config.yaml` and automatically loads the strongest
available model candidates in order.

If the weight package is shared through Google Drive, download it first and then
extract it from the project root:

```bash
unzip gridlock_model_weights_with_paths.zip -d .
```
