# Dataset Plan

For continued model improvement, build one YOLO-format dataset that contains
both road users and violation evidence objects:

- road users: `car`, `motorcycle`, `bus`, `truck`, `auto_rickshaw`, `bicycle`, `person`
- infrastructure: `traffic_light`, `stop_sign`, `stop_line`, `no_parking_sign`
- safety evidence: `helmet`, `no_helmet`, `seatbelt`, `no_seatbelt`
- OCR anchor: `license_plate`

Recommended labelling rules:

- Label the full violating vehicle, not only the rider, for final violation boxes.
- Label `no_helmet` around the visible head region when the rider head is visible and no helmet is present.
- Label `license_plate` tightly around the plate region.
- Keep a validation split with night, rain, glare, low-resolution, and crowded scenes.
- Add camera-specific calibration values in `backend/config.yaml` for stop line and wrong-side rules.

Training command:

```bash
python backend/scripts/train_detector.py \
  --data backend/datasets/traffic_violations.yaml \
  --model yolo26s.pt \
  --epochs 100 \
  --imgsz 960
```

After training, copy the best checkpoint to:

```text
backend/models/traffic_yolo26s_best.pt
```

The API will automatically prefer that file over generic pretrained weights.
