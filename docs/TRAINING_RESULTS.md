# Model And Training Summary

Gridlock AI Enforcer uses a trained YOLO traffic model with an additional helmet/plate specialist model. The trained weights are distributed separately from the source package.

## Runtime Model Files

```text
backend/models/traffic_yolo26s_best.pt
backend/models/helmet_violations_best.pt
backend/yolo26s.pt
backend/yolov3-tiny.weights
```

The backend loads candidates from `backend/config.yaml` in priority order. The trained traffic model is used first, then the specialist model, then fallback models.

## Training Data Sources

The local training pipeline supports:

- Kaggle helmet and plate datasets
- two-wheeler violation datasets
- Roboflow YOLO datasets for signs, red lights, parking, wrong-side examples, and related evidence classes
- manually curated judge-ready images for demonstration and regression testing

The training scripts normalize multiple formats into a YOLO dataset:

```text
backend/scripts/build_specialist_dataset.py
backend/scripts/build_multitask_dataset.py
backend/scripts/train_detector.py
```

## Model Classes

The multi-task dataset supports these project classes:

```text
helmet
no_helmet
license_plate
triple_riding
seatbelt
red_light
green_light
no_stopping_sign
no_entry_sign
stop_sign
```

The rule layer maps these detections into the required violation classes.

## Evaluation Workflow

Formal metrics are produced from labelled ground truth:

```bash
python backend/scripts/evaluate_dataset.py \
  --images path/to/eval/images \
  --ground-truth path/to/ground_truth.json \
  --dataset-name gridlock-val \
  --write-db
```

The evaluator calculates:

- Accuracy
- Precision
- Recall
- F1-score
- mAP50

The Performance tab reads the saved metrics and displays them with throughput and operational readiness.

## Current Verified Demo Regression

The curated local demo set currently verifies:

```text
Helmet Non-compliance: 5 images
Triple Riding: 3 images
License Plate OCR: 2 images
Seatbelt Non-compliance: 2 images
Illegal Parking: 1 image
```

The same set is packaged for ZIP batch upload:

```text
testing_usecases/gridlock_batch_upload_demo.zip
```

## Re-training Command

Example GPU training command:

```bash
python backend/scripts/train_detector.py \
  --data data/processed/traffic_multitask_detector_v2/dataset.yaml \
  --model backend/yolo26s.pt \
  --epochs 80 \
  --imgsz 960 \
  --batch -1 \
  --device 0 \
  --project backend/runs/train \
  --name gridlock_multitask_yolo26s
```

After training, copy the best checkpoint into the runtime slot:

```bash
cp runs/train/gridlock_multitask_yolo26s/weights/best.pt backend/models/traffic_yolo26s_best.pt
```

Then restart the backend and run the curated regression set again.
