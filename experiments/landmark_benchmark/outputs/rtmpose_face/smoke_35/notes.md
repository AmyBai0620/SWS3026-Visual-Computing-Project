# Notes

- Input: original FER-2013 48x48 grayscale image converted to three-channel BGR.
- Bounding box: the entire image, `[0, 0, width-1, height-1]`.
- The official MMPose pipeline performs the resize/affine transform to 256x256.
- Model loading, image reading, SHA-256 checking, CSV writing, and overlay saving are excluded from `inference_ms`.
- Warm-up calls are excluded from all timing statistics.
- `inference_success` only means that 106 finite keypoints were returned.
- It does **not** mean all landmarks are visually correct.
- Complete manual review in `sample_metrics_reviewed.csv` using:
  - `good`
  - `acceptable`
  - `wrong`
- Recommended `manual_valid`:
  - `1` for good or acceptable
  - `0` for wrong
