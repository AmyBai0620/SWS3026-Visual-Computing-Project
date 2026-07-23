# SWS3026-Visual-Computing-Project

Real-time facial expression effects and Just Dance pose analysis project for SWS3026 Visual Computing.

## 项目结构与讲稿索引

项目分三个层级，每层各有一份**实验结论 + 答辩讲稿**文档（对标评分细则 16 题）：

| 层级 | 目录 | 内容 | 讲稿（对应细则） |
| --- | --- | --- | --- |
| Beginner | `beginner/` | Haar + LBF 人脸检测/关键点，与 MediaPipe 的 10 场景鲁棒性对比 | [FINDINGS.md](beginner/FINDINGS.md)（第 1–3 题） |
| Expert | `expert/` | FER-2013 表情分类（LBF 68 点 + RBF-SVM）与实时特效 V2 | [PRESENTATION_NOTES.md](expert/PRESENTATION_NOTES.md)（第 4–10 题） |
| Bonus (Task 1) | `bonus/` | YOLOv8-pose 姿态流水线：主舞者选择、时序关联、EMA 平滑 | [FINDINGS.md](bonus/FINDINGS.md)（第 11–12 题） |
| Bonus (Task 2) | `bonus/` | Just Dance 评分系统 + 神庙逃亡体感游戏 | [TASK2_FINDINGS.md](bonus/TASK2_FINDINGS.md)（第 13–16 题） |

**运行环境**：conda 环境 `vcwork`（torch 2.7.1 CPU 版、opencv、mediapipe、scikit-learn、ultralytics）。系统 Python 没有 cv2，脚本请在 `vcwork` 下跑。

```bash
conda activate vcwork
python bonus/just_dance.py        # Just Dance 双面板（需先 precompute_reference.py）
python bonus/temple_run.py        # 体感神庙逃亡
python expert/realtime_demo_v2.py # 实时表情特效
```



