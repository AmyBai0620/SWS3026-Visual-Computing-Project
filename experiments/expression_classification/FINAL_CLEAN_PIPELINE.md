# Final cleaned expression-classification pipeline

## Objective

This final stage compares existing LBF and RTMPose arrays on one reproducible protocol: the **MediaPipe-filtered leakage-safe dataset**. Earlier official-unfiltered results are development-stage evidence and are not rerun.

```text
Raw FER-style dataset -> official manifest -> duplicate/leakage-safe manifest
-> MediaPipe Face Detection -> MediaPipe-filtered leakage-safe manifest
-> official-row subset indices -> StandardScaler + frozen Logistic Regression
-> final LBF versus RTMPose comparison
```

## Dataset and environment

Use a local dataset with `train/<class>/` and `test/<class>/` directories for angry, disgust, fear, happy, neutral, sad, and surprise. The dataset is intentionally not in Git.

```powershell
$DatasetRoot = 'path/to/local/facial_expression_dataset'
py -3.10 -m venv .venv-final-clean
.\.venv-final-clean\Scripts\Activate.ps1
python -m pip install -r experiments\expression_classification\requirements-final-clean.txt
```

## Commands from repository root

```powershell
# Official manifest and validation
python experiments\expression_classification\scripts\create_official_protocol_manifest.py --dataset-root $DatasetRoot --output-manifest experiments\expression_classification\manifests\official_protocol_manifest.csv --metadata-output experiments\expression_classification\manifests\official_protocol\metadata.json --seed 42 --overwrite
python experiments\expression_classification\scripts\validate_official_protocol_manifest.py --dataset-root $DatasetRoot --manifest experiments\expression_classification\manifests\official_protocol_manifest.csv --metadata experiments\expression_classification\manifests\official_protocol\metadata.json

# Duplicate audit and leakage-safe manifest
python experiments\expression_classification\scripts\audit_duplicate_images.py --dataset-root $DatasetRoot --output-report experiments\expression_classification\manifests\duplicate_audit.json --output-groups-csv experiments\expression_classification\manifests\duplicate_groups.csv --output-conflicts-csv experiments\expression_classification\manifests\duplicate_label_conflicts.csv
python experiments\expression_classification\scripts\create_full_manifest.py --dataset-root $DatasetRoot --output-manifest experiments\expression_classification\manifests\leakage_safe_manifest.csv --metadata-output experiments\expression_classification\manifests\leakage_safe\metadata.json --seed 42 --overwrite
python experiments\expression_classification\scripts\validate_manifest.py --dataset-root $DatasetRoot --manifest experiments\expression_classification\manifests\leakage_safe_manifest.csv --metadata experiments\expression_classification\manifests\leakage_safe\metadata.json

# MediaPipe screening: local raw_detections.csv is intentionally not committed
python experiments\expression_classification\scripts\run_mediapipe_face_validity.py --manifest experiments\expression_classification\manifests\official_protocol_manifest.csv --dataset-root $DatasetRoot --output-dir experiments\expression_classification\outputs\mediapipe_face_validity_official_v1 --checkpoint-every 250 --resume

# Final manifest/indexes and frozen final comparison
python experiments\expression_classification\scripts\build_mediapipe_filtered_leakage_safe_manifest.py --output-dir experiments\expression_classification\outputs\classifier_comparison_mediapipe_clean
python experiments\expression_classification\scripts\train_final_mediapipe_clean_comparison.py --output-dir experiments\expression_classification\outputs\classifier_comparison_mediapipe_clean
```

## Expected evidence

- Leakage-safe rows 35,111; MediaPipe-detected retained rows 32,960; exclusions 2,151.
- Cleaned splits: train 22,300; validation 3,927; test 6,733.
- LBF cleaned test: accuracy 0.4276, macro F1 0.3679.
- RTMPose cleaned test: accuracy 0.5415, macro F1 0.4784; improvements +11.39 accuracy points and +11.05 macro-F1 points.

Outputs are in `manifests/`, `outputs/mediapipe_face_validity_official_v1/`, and `outputs/classifier_comparison_mediapipe_clean/`.

## Limits and troubleshooting

Raw datasets, landmark arrays, raw MediaPipe detections, multi-signal tables, checkpoints, prediction tables, binary models, and complete contact sheets stay local. MediaPipe no-detection is not semantic-invalid ground truth: difficult faces can be excluded and non-face images can still be detected. Visual inspection informed failure-mode analysis, but no exhaustive human annotation of exclusions exists. If MediaPipe lacks `mp.solutions.face_detection`, recreate the Python 3.10 environment from the requirements file. The final trainer needs pre-existing official LBF and RTMPose arrays and does not rerun extraction.
