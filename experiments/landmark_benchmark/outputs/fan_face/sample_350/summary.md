# FAN 350-sample benchmark

## Automatic results

- Raw inference success: 350/350 (100.00%)
- Mean inference time: 173.67 ms
- Median inference time: 163.64 ms
- P95 inference time: 219.03 ms
- Maximum inference time: 249.10 ms
- Mean keypoint score: 0.7860
- Mean visible landmark rate: 98.92%
- Mean in-bounds landmark rate: 98.59%

## Manual review

- Good: 174
- Acceptable: 170
- Wrong: 6
- Manual valid: 344/350 (98.29%)
- Strict good rate: 49.71%

The review is conservative. Profile, severe crop, occlusion, blur, and inferred hidden-side landmarks are marked acceptable even when the output remains useful.

## Interpretation

FAN returned 68 finite landmarks for every sample, but the full FER image was supplied as an external face bounding box. High heatmap scores and in-bounds coordinates do not guarantee correct anatomical placement. The clearest failures occur under severe profile, hand or hair occlusion, and heavy blur.
