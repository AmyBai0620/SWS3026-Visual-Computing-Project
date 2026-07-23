# Expert Level — Final Expression System

## Final runtime files

| File | Purpose |
|---|---|
| `run_emotion_camera.py` | Canonical final entry point |
| `realtime_demo_v10_threaded_scrapbook.py` | Scrapbook composition and V10 window entry |
| `realtime_demo_v10_threaded.py` | Main camera, detection, animation, smoothing, and UI loop |
| `async_expression_worker.py` | Single RTMPose worker with a latest-frame single-slot buffer |
| `rtmpose_emotion_recognizer.py` | RTMPose landmarks, feature construction, scaling, and classification |
| `robust_face_detector.py` | MediaPipe detector, Haar fallback, smoothing, and short hold |
| `prediction_types.py` | Shared prediction result dataclass |
| `emotion_effects.py` | Common seven-emotion effects |
| `happy_effect_renderer_v8f.py` | Final Happy renderer |
| `surprise_effect_renderer_v8f.py` | Final Surprise renderer |
| `wide_screen_emotion_renderer_v8f.py` | Final Angry/Disgust/Fear/Neutral/Sad wide-screen renderer |
| `scrapbook_sidebar.py` | Final scrapbook sidebar |

## Runtime data

```text
models/rtmpose_expression/     classifier.joblib + scaler.joblib
models/rtmpose_face/           RTMPose-Face checkpoint
assets/effects/<emotion>/      manifest.json + runtime PNG assets
assets/ui/sidebar/             paper, titles, strips, and clouds
```

## Architecture

The OpenCV main thread handles camera capture, face detection, animation, sidebar drawing, display, and keyboard input. Exactly one background thread owns the recognizer. New face crops overwrite the single pending slot, so stale frames never build up in a queue.

This preserves the animation while preventing RTMPose from blocking every display frame.

## Diagnostics

- `benchmark_expression_prediction.py`: fixed-input prediction benchmark.
- `realtime_rtmpose_classifier_test.py`: robust detector, landmarks, classification, and live timing.
- `check_rtmpose_model.py`: classifier/scaler validation.
- `check_rtmpose_expression_pipeline.py`: end-to-end file and pipeline checks.
- `check_scrapbook_ui_assets.py`: seven-sidebar render check.
