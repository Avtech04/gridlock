# Feature Coverage

This document maps the hackathon requirement to the implemented system.

## Requirement Map

| Requirement | Implementation | Main Files |
|---|---|---|
| Image preprocessing | Brightness/contrast analysis, CLAHE illumination normalization, denoising, and sharpening | `backend/main.py` |
| Low light, rain, shadows, blur | Preprocessing pipeline plus curated test cases | `backend/main.py`, `testing_usecases/` |
| Vehicle detection | YOLO ensemble detects cars, motorcycles, buses, trucks, bicycles, and fallback vehicle classes | `backend/detector.py` |
| Road-user detection | Person/pedestrian detection and rider association with motorcycles | `backend/detector.py`, `backend/violation_mapper.py` |
| Helmet non-compliance | Helmet and no-helmet specialist detections anchored to riders/motorcycles | `backend/detector.py`, `backend/violation_mapper.py` |
| Seatbelt non-compliance | No-seatbelt class support plus close-cabin occupant rule | `backend/violation_mapper.py` |
| Triple riding | Direct triple-riding evidence and rider-count association | `backend/violation_mapper.py` |
| Wrong-side driving | Direct class support, lane-arrow evidence, and calibration support | `backend/violation_mapper.py`, `backend/video_analyzer.py` |
| Stop-line violation | Stop-line class support, road-marking detection, and calibration gate | `backend/violation_mapper.py` |
| Red-light violation | Traffic-light state detection plus stop-line/gate logic | `backend/detector.py`, `backend/violation_mapper.py` |
| Illegal parking | Direct class support, no-parking sign inference, and configurable ROI | `backend/detector.py`, `backend/violation_mapper.py` |
| Violation classification | Normalized required violation classes with confidence scores | `backend/violation_mapper.py` |
| License plate detection | YOLO plate detection plus OpenCV plate candidates | `backend/detector.py`, `backend/ocr.py` |
| License plate OCR | Tesseract by default; PaddleOCR/EasyOCR optional | `backend/ocr.py` |
| Annotated evidence | Bounding boxes, labels, plates, confidence scores, and timestamps | `backend/main.py` |
| Metadata storage | SQLite stores records, evidence paths, review status, timestamps, plates, and bboxes | `backend/main.py` |
| Searchable records | Records tab and `/api/violations` filtering/search | `backend/main.py`, `web/` |
| Analytics and reporting | Violation totals, trends, top plates, class distribution, and AI/local command brief | `backend/main.py`, `web/` |
| Performance evaluation | Accuracy, Precision, Recall, F1, and mAP workflow from labelled ground truth | `backend/evaluator.py`, `backend/scripts/evaluate_dataset.py` |
| Scalability | ZIP batch upload, per-image batch results, review approval, and upload limits | `backend/main.py`, `web/` |

## Verified Demo Set

The completed local demo set is available in:

```text
testing_usecases/judge_ready/
testing_usecases/gridlock_batch_upload_demo.zip
```

The batch zip contains 13 verified images across:

- Helmet Non-compliance
- Triple Riding
- License Plate OCR
- Seatbelt Non-compliance
- Illegal Parking

## Evaluation Method

The dashboard is designed to show measured metrics only after labelled ground truth is supplied. Run:

```bash
python backend/scripts/evaluate_dataset.py \
  --images path/to/eval/images \
  --ground-truth path/to/ground_truth.json \
  --dataset-name gridlock-val \
  --write-db
```

The Performance page then displays the stored evaluation results.

## Notes For Reviewers

- The source package is intentionally separate from model weights.
- The model weights package should be extracted from the project root before running the best demo.
- Single-image workflows and ZIP batch workflows are both implemented.
- Optional Gemini/Groq integration improves narrative analytics, but the core detector, OCR, evidence generation, records, and reports run without external API keys.
