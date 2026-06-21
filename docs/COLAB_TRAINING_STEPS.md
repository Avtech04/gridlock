# Optional Colab Training Guide

This guide is optional. It is useful when retraining the traffic detector on a GPU machine.

## Goal

Train a multi-task YOLO detector and copy the best checkpoint back into:

```text
backend/models/traffic_yolo26s_best.pt
```

## Recommended Colab Runtime

1. Open Google Colab.
2. Select `Runtime -> Change runtime type -> GPU`.
3. Upload the prepared training package if available:

```text
gridlock_colab_training_package_v3.zip
```

4. Extract the package in Colab.

## Training Command

```python
from ultralytics import YOLO

model = YOLO("/content/gridlock/backend/yolo26s.pt")
model.train(
    data="/content/gridlock/data/processed/traffic_multitask_detector_v2/dataset.yaml",
    epochs=80,
    imgsz=960,
    batch=-1,
    device=0,
    project="/content/drive/MyDrive/gridlock_runs",
    name="gridlock_multitask_yolo26s",
    patience=20,
    cos_lr=True,
    close_mosaic=15,
    multi_scale=0.15,
    cache=True,
    plots=True,
)
```

If the GPU runs out of memory, use:

```text
imgsz=768
batch=4
```

## Export Back To Project

After training, download:

```text
/content/drive/MyDrive/gridlock_runs/gridlock_multitask_yolo26s/weights/best.pt
```

Rename it locally to:

```text
traffic_yolo26s_best.pt
```

Place it here:

```text
backend/models/traffic_yolo26s_best.pt
```

Restart the backend and run the curated test package:

```text
testing_usecases/gridlock_batch_upload_demo.zip
```
