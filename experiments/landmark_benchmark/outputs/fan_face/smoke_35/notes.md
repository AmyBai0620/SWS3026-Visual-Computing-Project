# Notes

- Input: original FER-2013 48x48 grayscale image converted to three-channel RGB.
- Bounding box: the entire image, `[0, 0, width-1, height-1]`.
- FAN's internal face detector is bypassed by passing `detected_faces`.
- `flip_input=False` and `compile=False`.
- Model loading, image reading, SHA-256 checking, CSV writing, and overlay saving are excluded from `inference_ms`.
- Warm-up calls are excluded from all timing statistics.
- `inference_success` only means that 68 finite keypoints were returned.
- It does **not** mean all landmarks are visually correct.
- FAN heatmap scores may remain high even when eyebrows or occluded-side landmarks are visibly misplaced.
- Complete manual review in `sample_metrics_reviewed.csv` using:
  - `good`
  - `acceptable`
  - `wrong`
- Recommended `manual_valid`:
  - `1` for good or acceptable
  - `0` for wrong
