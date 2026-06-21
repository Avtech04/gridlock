# Gridlock AI Testing Use Cases

Use these files from the dashboard at:

```text
http://localhost:8000/ui/
```

## Judge-Ready Single Images

| Folder | Purpose |
|---|---|
| `judge_ready/helmet_non_compliance/` | Five real-world two-wheeler images with visible no-helmet evidence |
| `judge_ready/triple_riding/` | Three clear triple-riding examples |
| `judge_ready/license_plate_ocr/` | Two plate OCR examples: `CZ20FSE` and `29A33185` |
| `judge_ready/seatbelt_non_compliance/` | Two close-cabin no-seatbelt examples |
| `judge_ready/illegal_parking/` | Visible no-parking sign with a parked vehicle |

## ZIP Batch Demo

| Path | Purpose |
|---|---|
| `batch_upload_demo/` | Flat folder containing the same verified images |
| `gridlock_batch_upload_demo.zip` | Upload this directly in the UI's ZIP batch mode |

The batch ZIP contains 13 images and is intended for a fast end-to-end
demonstration of image upload, batch processing, annotated evidence generation,
records, and analytics.

## Suggested Demo Flow

1. Start the backend with `./start.sh`.
2. Open `http://localhost:8000/ui/`.
3. Upload one image from `judge_ready/helmet_non_compliance/`.
4. Upload one image from `judge_ready/license_plate_ocr/`.
5. Switch to ZIP batch mode.
6. Upload `gridlock_batch_upload_demo.zip`.
7. Open Records to show stored evidence images.
8. Open Analytics and Performance for summary reporting.
