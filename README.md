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

---

## 实时表情识别 V2 更新说明（2026.7.19）

在保留原始 `expert/realtime_demo.py` 的基础上，新增：

```bash
python expert/realtime_demo_v2.py
````

V2 版本主要完成了实时表情识别结果的后处理与展示功能。

### 主要更新

* 新增主脸筛选，只保留画面中的主要人脸，减少同一张脸出现多个嵌套检测框的问题；
* 新增轻量级时间平滑，降低单帧误判造成的标签频繁跳变；
* 将关键点提取、特征处理和 SVM 分类封装到独立的 `emotion_recognizer.py` 中；
* 新增统一的 `EmotionPrediction` 输出接口，便于后续替换新的关键点模型和分类器；
* 新增 `emotion_effects.py`，为 FER-2013 的七类表情提供不同的动态视觉效果；
* 新增 `emotion_stickers.py`，支持透明 PNG 贴纸的加载、缓存、缩放和 Alpha 混合；
* 为 Angry、Disgust、Fear、Happy、Neutral、Sad 和 Surprise 七类表情配置独立的主贴纸、结果徽章和装饰素材；
* 新增底部轻量状态栏，显示检测结果、展示结果、运行模式、置信度和平均单帧耗时；
* 新增手动预览模式，可在不依赖分类器预测结果的情况下检查每种表情特效。

### 相关文件

```text
expert/
├── realtime_demo.py          # 原始实时识别版本
├── realtime_demo_v2.py       # V2 特效展示版本
├── emotion_recognizer.py     # 关键点提取与表情分类接口
├── emotion_effects.py        # 动态特效渲染
├── emotion_stickers.py       # 透明 PNG 贴纸渲染
└── assets/
    └── effects/              # 七类表情贴纸及装饰素材
```

### 操作方式

```text
1：预览 Angry 特效
2：预览 Disgust 特效
3：预览 Fear 特效
4：预览 Happy 特效
5：预览 Neutral 特效
6：预览 Sad 特效
7：预览 Surprise 特效
0：返回自动识别模式
E：开启或关闭特效
Q：退出程序
```

手动预览模式只会覆盖当前展示的特效，不会停止后台分类器运行，因此可以同时查看真实识别结果和指定表情的视觉效果。

### 当前限制

当前 V2 仍然使用原始的 Haar + LBF + SVM 表情分类流程，因此实时识别中仍可能出现以下问题：

* Neutral、Sad 和 Angry 之间容易混淆；
* Surprise 和 Fear 之间容易混淆；
* 侧脸、遮挡、近距离和低清画面下的 LBF 关键点可能不稳定；
* 当前旧 SVM 模型未提供有效的分类概率，因此置信度可能显示为 `N/A`。

后续新的关键点模型和分类器训练完成后，可以通过 `emotion_recognizer.py` 接入现有 V2 框架，无需重新实现摄像头界面、动态特效和贴纸渲染。


