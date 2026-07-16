# RTMPose-Face 350-sample benchmark

## Automatic results

- Raw inference success: 350/350 (100.00%)
- Mean inference time: 108.29 ms
- Median inference time: 107.03 ms
- P95 inference time: 122.91 ms
- Maximum inference time: 138.37 ms
- Mean keypoint confidence: 0.6602
- Mean in-bounds landmark rate: 91.39%

## Manual review

- Good: 155
- Acceptable: 195
- Wrong: 0
- Manual valid: 350/350 (100.00%)
- Strict good rate: 44.29%

The review is deliberately conservative: profile, severe crop, occlusion, low contrast, and inferred hidden-side landmarks are marked acceptable rather than good, even when the result remains useful for expression features.

## Interpretation

RTMPose-Face returned 106 finite landmarks for every image, but this is partly because the full FER image was supplied as a face bounding box. The model can therefore predict a complete landmark set even when part of the face is hidden or outside the image. Manual labels and the in-bounds rate should be reported alongside raw inference success.
