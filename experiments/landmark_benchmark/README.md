# FER-2013 人脸关键点模型小样本测试  
## 交付与统一说明文档

> 适用模型：MediaPipe、RTMPose-Face、FAN  
> 当前阶段目标：先比较关键点模型在 FER-2013 低分辨率图片上的可行性，再决定是否值得跑全量并重新训练表情分类器。

---

## 1. 测试目的

本轮小样本测试**不是用来训练最终分类器**，而是先回答下面几个问题：

1. 模型能否在 FER-2013 的 48×48 低分辨率图片上稳定提取关键点；
2. 关键点是否真正贴合眼睛、眉毛、鼻子和嘴部；
3. 张嘴时嘴部关键点是否有明显响应；
4. 是否存在“程序返回成功，但关键点位置错误”的情况；
5. 是否出现不同图片输出过于相似的模板化结果；
6. 单张图片处理时间是否适合后续全量提取和实时接入；
7. 安装、运行和接入现有项目的难度是否可接受。

本轮结束后，三种模型中表现较好的 1～2 个才进入全量特征提取和分类器训练阶段。

---

## 2. 统一样本

所有人必须使用同一批样本，不允许各自重新随机抽取。

### 2.1 35 张试跑集

- 每类 5 张；
- 共 7 类，合计 35 张；
- 用于确认环境、代码、输入输出格式和基本关键点质量；
- 文件：`sample_manifest_35.csv`。

### 2.2 350 张正式小样本

- 每类 50 张；
- 共 7 类，合计 350 张；
- 用于比较成功率、质量、耗时和失败情况；
- 文件：`sample_manifest_350.csv`。

### 2.3 类别顺序

```text
angry
disgust
fear
happy
neutral
sad
surprise
```

### 2.4 抽样规则

- 数据来源：FER-2013 `train`；
- 随机种子：42；
- 35 张集合是 350 张集合的子集；
- manifest 中包含相对路径和 SHA-256；
- 每个人只修改自己本地的 `DATASET_ROOT`；
- 不上传原始 FER 图片。

---

## 3. 推荐目录结构

```text
experiments/
└── landmark_benchmark/
    ├── README.md
    ├── create_sample_manifest.py
    ├── manifests/
    │   ├── sample_manifest_35.csv
    │   ├── sample_manifest_350.csv
    │   └── manifest_metadata.json
    ├── runners/
    │   ├── mediapipe_runner.py
    │   ├── rtmpose_face_runner.py
    │   └── fan_runner.py
    └── outputs/
        ├── mediapipe/
        │   ├── smoke_35/
        │   └── sample_350/
        ├── rtmpose_face/
        │   ├── smoke_35/
        │   └── sample_350/
        └── fan/
            ├── smoke_35/
            └── sample_350/
```

---

## 4. 三个人必须统一的输入

每个 runner 至少接收：

```text
--dataset-root
--manifest
--output-dir
```

示例：

```powershell
python .\runners\mediapipe_runner.py `
  --dataset-root "E:\...\facial_expression_dataset\train" `
  --manifest ".\manifests\sample_manifest_35.csv" `
  --output-dir ".\outputs\mediapipe\smoke_35"
```

runner 必须按照 manifest 的 `relative_path` 读取图片，不允许自行重新抽样。

---

## 5. 统一输出结构

### 5.1 35 张试跑输出

```text
outputs/<model_name>/smoke_35/
├── sample_metrics.csv
├── summary.json
├── notes.md
├── environment.md
└── overlays/
    ├── angry_001.jpg
    ├── angry_002.jpg
    └── ...
```

35 张必须全部保存可视化结果并人工检查。

### 5.2 350 张正式小样本输出

```text
outputs/<model_name>/sample_350/
├── sample_metrics.csv
├── summary.json
├── summary.md
├── environment.md
├── success_examples/
└── failure_examples/
```

350 张全部自动统计；人工检查时，每类至少抽 5 张，共 35 张。

---

## 6. `sample_metrics.csv` 统一字段

建议统一为：

```csv
sample_id,label,relative_path,extract_success,num_landmarks,runtime_ms,landmark_valid,eye_fit,eyebrow_fit,mouth_fit,mouth_open_response,template_risk,failure_reason
```

字段说明：

| 字段 | 含义 |
|---|---|
| `sample_id` | manifest 中的统一样本 ID |
| `label` | FER 表情类别 |
| `relative_path` | 相对数据集根目录的路径 |
| `extract_success` | 是否成功返回关键点，填 `true/false` |
| `num_landmarks` | 实际返回的关键点数量 |
| `runtime_ms` | 单张图片关键点提取耗时 |
| `landmark_valid` | 关键点整体是否合理 |
| `eye_fit` | 眼睛区域贴合质量 |
| `eyebrow_fit` | 眉毛区域贴合质量 |
| `mouth_fit` | 嘴部区域贴合质量 |
| `mouth_open_response` | 张嘴时嘴部点是否真实分开 |
| `template_risk` | 是否疑似输出固定模板 |
| `failure_reason` | 失败或异常原因 |

### 6.1 人工质量字段统一取值

以下字段统一使用：

```text
good
acceptable
wrong
not_applicable
```

适用于：

- `landmark_valid`
- `eye_fit`
- `eyebrow_fit`
- `mouth_fit`
- `mouth_open_response`

`template_risk` 统一使用：

```text
low
medium
high
unknown
```

---

## 7. “成功”与“有效”必须分开

不能只统计模型有没有返回关键点。

### 7.1 提取成功

```text
extract_success = true
```

表示程序返回了关键点。

### 7.2 关键点有效

```text
landmark_valid = good / acceptable
```

表示关键点位置与真实五官基本一致，可以用于后续特征提取。

例如下面这些情况应视为无效：

- 点明显落在脸外；
- 眼睛、嘴巴位置错位；
- 侧脸时仍输出一套正脸模板；
- 遮挡时大量点落到手上；
- 不同图片输出几乎相同的关键点布局；
- 程序返回成功，但关键点没有表达张嘴或眉眼变化。

---

## 8. 统一统计指标

每个模型必须至少输出：

### 8.1 基础统计

- 总样本数；
- 提取成功数；
- 提取成功率；
- 有效关键点数；
- 有效关键点率；
- 每类提取成功率；
- 每类有效关键点率。

### 8.2 运行时间

- 平均耗时；
- 中位数耗时；
- P95 耗时；
- 最慢单张耗时；
- 测试设备与环境。

### 8.3 人工质量

- 眼睛贴合情况；
- 眉毛贴合情况；
- 嘴部贴合情况；
- 张嘴响应；
- 模板化风险；
- 典型成功案例；
- 典型失败案例。

### 8.4 工程难度

- 安装是否顺利；
- 是否需要 GPU；
- 模型大小；
- CPU 运行是否可接受；
- Windows 环境兼容性；
- 接入实时摄像头的难度；
- 接入现有分类特征流程的难度。

---

## 9. `summary.json` 统一格式

```json
{
  "model_name": "MediaPipe",
  "model_version": "",
  "total_samples": 350,
  "extract_success_count": 0,
  "extract_success_rate": 0.0,
  "landmark_valid_count": 0,
  "landmark_valid_rate": 0.0,
  "num_landmarks": 0,
  "mean_runtime_ms": 0.0,
  "median_runtime_ms": 0.0,
  "p95_runtime_ms": 0.0,
  "max_runtime_ms": 0.0,
  "per_class_results": {},
  "installation_difficulty": "",
  "realtime_integration_difficulty": "",
  "hardware": "",
  "software_environment": "",
  "recommend_full_run": false,
  "notes": ""
}
```

---

## 10. `environment.md` 必须记录

```text
操作系统：
Python 版本：
模型名称与版本：
主要依赖版本：
CPU：
GPU：
是否使用 GPU：
模型权重来源：
安装过程中遇到的问题：
运行命令：
```

不同电脑的耗时不能直接作为最终横向结论。最终速度比较要在同一台演示电脑上重跑。

---

## 11. 第一轮：35 张试跑交付要求

每个人必须交：

1. 可运行的 runner；
2. `sample_metrics.csv`；
3. 35 张关键点可视化；
4. `summary.json`；
5. `environment.md`；
6. `notes.md`；
7. 是否建议继续跑 350 张的结论。

### 完成标准

- 35 张全部按 manifest 读取；
- 每张都有对应输出记录；
- 可视化图能清楚看到关键点；
- 报错样本也必须记录，不能静默跳过；
- 运行命令可复现；
- 不覆盖原始图片和原 baseline 文件。

---

## 12. 第二轮：350 张正式小样本交付要求

每个人必须交：

1. 350 张完整 `sample_metrics.csv`；
2. `summary.json`；
3. `summary.md`；
4. 各类别提取成功率；
5. 各类别有效关键点率；
6. 平均、中位数和 P95 耗时；
7. 每类典型成功图 2～3 张；
8. 每类典型失败图 2～3 张；
9. 是否推荐全量运行的明确结论；
10. 已知风险和接入难点。

---

## 13. 三种模型最终对比表

最终由一人统一汇总：

| 对比项 | MediaPipe | RTMPose-Face | FAN |
|---|---:|---:|---:|
| 35 张是否跑通 |  |  |  |
| 350 张提取成功率 |  |  |  |
| 350 张有效关键点率 |  |  |  |
| 眼睛贴合 |  |  |  |
| 眉毛贴合 |  |  |  |
| 嘴部贴合 |  |  |  |
| 张嘴响应 |  |  |  |
| 模板化风险 |  |  |  |
| 平均耗时 |  |  |  |
| 中位数耗时 |  |  |  |
| P95 耗时 |  |  |  |
| 关键点数量 |  |  |  |
| 是否需要 GPU |  |  |  |
| 安装难度 |  |  |  |
| 实时接入难度 |  |  |  |
| 是否推荐跑全量 |  |  |  |

---

## 14. 分类效果不在本轮小样本中直接比较

35 张和 350 张只用于筛选关键点模型，不用于训练最终分类器。

筛选出表现最好的 1～2 个候选后，再统一：

1. 划分 train / validation；
2. 保持原 test 不参与调参；
3. 全量提取关键点特征；
4. 使用统一的特征归一化原则；
5. 第一轮使用同一种 RBF SVM；
6. 使用相同的参数搜索范围；
7. 比较 Validation Accuracy、Macro F1 和各类 Recall；
8. 确定最终方案后再在 test 上评估一次；
9. 最后比较实时速度和标签稳定性。

---

## 15. 本阶段不上传的内容

- 原始 FER 数据集；
- 个人摄像头视频；
- 大量临时图片；
- 模型缓存；
- 未完成的权重文件；
- 环境中的虚拟环境文件夹；
- 中间备份模型；
- 含绝对本地路径的私人配置文件。

manifest、runner、统计结果和少量代表性失败案例可以上传仓库。

---

## 16. 建议分工

- A：MediaPipe runner 和结果；
- B：RTMPose-Face runner 和结果；
- C：FAN runner 和结果；
- D：统一 manifest、字段规范、结果汇总和复核。

负责 manifest 的人不重新抽样；manifest 一旦上传，后续不能随意修改。

---

## 17. 本轮最终决策规则

优先选择：

1. 在 FER 小图上有效关键点率更高；
2. 眼睛、眉毛和嘴部贴合更可靠；
3. 张嘴响应更明显；
4. 模板化风险更低；
5. 运行时间可接受；
6. 安装和实时接入风险可控。

如果两个模型结果接近，则都进入全量分类实验，不能只根据小样本中 1%～2% 的差异直接下结论。



## RTMPose-Face

### Model and setup

- Model: RTMPose-M Face6
- Number of landmarks: 106
- Input size: 256 × 256
- FER-2013 input size: 48 × 48 grayscale
- The whole FER image was used as the face bounding box
- Benchmark device: CPU
- MMPose version: 1.3.2

### 35-sample smoke test

- Raw inference success: 35/35 (100.00%)
- Good: 20
- Acceptable: 15
- Wrong: 0
- Manual valid rate: 100.00%
- Mean inference time: 105.70 ms
- Median inference time: 103.94 ms
- P95 inference time: 115.21 ms

### 350-sample benchmark

- Raw inference success: 350/350 (100.00%)
- Good: 155
- Acceptable: 195
- Wrong: 0
- Manual valid rate: 100.00%
- Strict good rate: 44.29%
- Mean inference time: 108.29 ms
- Median inference time: 107.03 ms
- P95 inference time: 122.91 ms
- Maximum inference time: 138.37 ms
- Mean keypoint confidence: 0.6602
- Mean in-bounds landmark rate: 91.39%

RTMPose-Face returned 106 finite landmarks for every sample. However,
because the whole image was supplied as the face bounding box, raw inference
success does not necessarily mean that every predicted landmark is correct.

For profile, occluded, or tightly cropped faces, RTMPose-Face may estimate
landmarks for hidden or out-of-frame facial regions. Therefore, manual review
and the in-bounds landmark rate should be reported together with raw inference
success.


## Preliminary landmark model comparison

| Metric | MediaPipe Face Mesh | RTMPose-Face |
|---|---:|---:|
| Number of landmarks | 468 | 106 |
| Raw success | 317/350 | 350/350 |
| Good | 150 | 155 |
| Acceptable | 167 | 195 |
| Wrong | 33 | 0 |
| Manual valid rate | 90.57% | 100.00% |
| Mean inference time | 4.86 ms | 108.29 ms |
| Median inference time | 4.86 ms | 107.03 ms |
| P95 inference time | 6.06 ms | 122.91 ms |

This comparison is preliminary. FAN should be evaluated using the same
manifest before selecting the final landmark model.
