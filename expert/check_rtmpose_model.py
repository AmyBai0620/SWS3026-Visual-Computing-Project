from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

EXPECTED_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
EXPECTED_FEATURES = 318


def main() -> int:
    expert_dir = Path(__file__).resolve().parent
    model_dir = expert_dir / "models" / "rtmpose_expression"
    classifier_path = model_dir / "classifier.joblib"
    scaler_path = model_dir / "scaler.joblib"

    print("=" * 64)
    print("RTMPose expression model check")
    print("=" * 64)
    print(f"Python:        {sys.version.split()[0]}")
    print(f"NumPy:         {np.__version__}")
    print(f"scikit-learn:  {sklearn.__version__}")
    print(f"Model folder:  {model_dir}")

    missing = [p for p in (classifier_path, scaler_path) if not p.is_file()]
    if missing:
        print("\n[FAIL] Missing model file(s):")
        for path in missing:
            print(f"  - {path}")
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = joblib.load(scaler_path)
        classifier = joblib.load(classifier_path)

    classes = [str(x) for x in classifier.classes_.tolist()]
    classifier_features = int(classifier.n_features_in_)
    scaler_features = int(scaler.n_features_in_)

    print(f"\nClassifier:    {type(classifier).__name__}")
    print(f"Scaler:        {type(scaler).__name__}")
    print(f"Classes:       {classes}")
    print(f"Input dims:    classifier={classifier_features}, scaler={scaler_features}")

    checks = {
        "class order": classes == EXPECTED_CLASSES,
        "classifier feature count": classifier_features == EXPECTED_FEATURES,
        "scaler feature count": scaler_features == EXPECTED_FEATURES,
    }

    probe = np.zeros((1, EXPECTED_FEATURES), dtype=np.float64)
    transformed = scaler.transform(probe)
    prediction = str(classifier.predict(transformed)[0])
    probabilities = classifier.predict_proba(transformed)[0]

    checks["transformed shape"] = transformed.shape == (1, EXPECTED_FEATURES)
    checks["finite transformed values"] = bool(np.isfinite(transformed).all())
    checks["probability count"] = probabilities.shape == (len(EXPECTED_CLASSES),)
    checks["probability sum"] = bool(np.isclose(probabilities.sum(), 1.0))

    print("\nProbe result:")
    print(f"  prediction:  {prediction}")
    print(f"  proba shape: {probabilities.shape}")
    print(f"  proba sum:   {probabilities.sum():.6f}")

    print("\nChecks:")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    if caught:
        print("\nCompatibility warning(s):")
        for item in caught:
            print(f"  - {item.message}")

    if not all(checks.values()):
        print("\n[FAIL] Model files loaded, but one or more validations failed.")
        return 2

    print("\n[PASS] Final RTMPose classifier and scaler are ready for integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
