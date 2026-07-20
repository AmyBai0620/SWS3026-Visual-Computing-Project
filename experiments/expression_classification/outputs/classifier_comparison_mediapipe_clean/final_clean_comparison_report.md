# Final cleaned LBF vs RTMPose comparison

Only the final **MediaPipe-filtered leakage-safe protocol** was used for this final training and evaluation. Duplicate/leakage cleaning was applied first; MediaPipe Face Detection then constructed the reproducible face-detected subset. Earlier unfiltered outputs remain development-stage evidence and were not re-evaluated here.

## Protocol and class balance

- Cleaned rows: 32,960 — train 22,300; validation 3,927; test 6,733.
- Final class totals: angry 4,360; disgust 418; fear 4,586; happy 8,629; neutral 5,957; sad 5,448; surprise 3,562.
- Frozen model for each representation: `StandardScaler` fit on cleaned train only, then `LogisticRegression(C=1, class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=42)`.

## Results

| Method | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|---:|---:|
| LBF (68 points, 136 features) | 0.4217 | 0.3632 | 0.4276 | 0.3679 | 0.4274 |
| RTMPose (106 points + confidence, 318 features) | 0.5343 | 0.4638 | 0.5415 | 0.4784 | 0.5463 |

RTMPose improves cleaned-test accuracy by **+0.1139** (11.39 percentage points), macro F1 by **+0.1105**, macro precision by **+0.1092**, macro recall by **+0.1074**, and weighted F1 by **+0.1188** over LBF.

RTMPose test F1 exceeds LBF for every class: angry 0.4504 vs 0.2824; disgust 0.1990 vs 0.1469; fear 0.2828 vs 0.1828; happy 0.7989 vs 0.6615; neutral 0.5114 vs 0.4325; sad 0.4266 vs 0.3147; surprise 0.6795 vs 0.5543.

## Error patterns and interpretation

The main remaining confusions are sad/neutral and fear with sad, surprise, or neutral. For example, RTMPose misclassified 259 sad test samples as neutral and 193 neutral samples as sad; fear was often predicted as sad (205) or surprise (139). Happy and surprise remain stronger classes. Disgust has only 96 cleaned test examples and 418 cleaned examples total, so its recall/F1 and apparent gains should be interpreted cautiously.

RTMPose is selected as the final landmark representation: it is clearly stronger on the final cleaned test metrics. It remains slower at landmark extraction than LBF, so real-time deployment should measure end-to-end latency separately.

## Limitations

MediaPipe no-detection does not prove that an excluded image is semantically invalid: difficult, blurred, profile, dark, occluded, or partial faces may be filtered. Conversely, some semantic non-face images can still pass face detection. The protocol prioritizes simplicity, reproducibility, duplicate/leakage cleaning, and cleaner inputs; it is not a complete semantic-validity ground truth.
