# Real-time Facial Expression Video Effects

本部分从**已训练好的表情分类器接入**开始，完成了实时人脸检测、RTMPose-Face 关键点提取、表情分类、结果平滑、七类动态视频特效以及手帐风侧栏 UI 的整合。最终版本采用 **V10 单推理线程架构**，在保留全部动画效果的同时，避免 RTMPose 推理阻塞主界面。

## 1. 最终运行入口

在项目根目录运行：

```powershell
python expert\realtime_demo_v10_threaded_scrapbook.py
```

成功启动后，窗口标题应为：

```text
RTMPose Emotion Camera v10 - Threaded Scrapbook UI
```

## 2. 最终处理流程

```text
Webcam frame
    ↓
RobustFaceDetector
(MediaPipe FaceMesh primary + Haar fallback)
    ↓
Smoothed face bounding box and face crop
    ↓
Single-slot asynchronous inference worker
    ↓
RTMPose-Face: 106 facial keypoints + 106 confidence scores
    ↓
318-dimensional normalized landmark feature
    ↓
StandardScaler
    ↓
Trained Logistic Regression classifier
    ↓
Seven-class probability and confidence
    ↓
Temporal voting and label stabilization
    ↓
Emotion-specific animated effects
    ↓
Scrapbook sidebar and final OpenCV display
```

支持的七种表情类别为：

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

## 3. 接入已训练好的分类器

最终实时识别器由 `RTMPoseEmotionRecognizer` 实现，加载以下文件：

```text
expert/models/rtmpose_expression/classifier.joblib
expert/models/rtmpose_expression/scaler.joblib
experiments/landmark_benchmark/models/rtmpose_face/
    rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth
```

分类器不是直接使用整张人脸图像，而是使用 RTMPose-Face 输出的面部关键点特征。

每张人脸经过以下步骤：

1. RTMPose-Face 提取 `106` 个二维面部关键点；
2. 同时保留每个关键点的 `106` 个置信度分数；
3. 将关键点坐标减去所有点的中心位置，消除平移影响；
4. 使用关键点到中心的平均距离进行尺度归一化；
5. 将 `212` 个归一化坐标与 `106` 个置信度拼接；
6. 得到总长度为 `318` 的特征向量；
7. 使用 `StandardScaler` 标准化；
8. 使用训练好的 Logistic Regression 执行一次 `predict_proba()`；
9. 选择最高概率类别作为当前原始表情，并将其概率作为置信度。

```text
106 × 2 coordinate values + 106 confidence scores = 318 features
```

代码只调用一次 `predict_proba()`，再通过最大概率索引确定类别，避免同时调用 `predict()` 和 `predict_proba()` 所产生的重复计算。

## 4. 稳健人脸检测

最终版本使用 `RobustFaceDetector`，而不是只依赖 Haar Cascade。

检测流程为：

```text
MediaPipe FaceMesh
    ↓ failed
Haar Cascade fallback
    ↓
Bounding-box smoothing
    ↓
Short-term box holding when detection is temporarily lost
```

具体设计：

- MediaPipe FaceMesh 作为主要检测与跟踪模块；
- Haar Cascade 只在 MediaPipe 暂时失败时使用；
- 根据 FaceMesh 面部表面关键点计算完整人脸裁剪区域；
- 对检测框的位置和尺寸进行平滑，减少轻微抖动；
- 短时间检测失败时继续保持上一帧的人脸框；
- 当前实现最多保持 `6` 帧，避免遮挡或快速转头时特效立即消失。

这一结构比单独使用 Haar 更适合侧脸、轻度遮挡、距离变化和头部转动。

## 5. 表情结果稳定处理

逐帧分类结果可能因为光照、头部姿态和关键点噪声而短暂跳变。最终程序使用时间窗口投票进行平滑：

```text
SMOOTH_WINDOW = 5
MIN_VOTES = 3
RESET_AFTER_MISSING = 8
```

处理逻辑：

- 保存最近 `5` 次有效预测；
- 某个类别至少获得 `3` 票后才更新稳定标签；
- 人脸连续丢失 `8` 帧后，清空历史结果并重置动画；
- 原始标签仍可在调试模式中查看。

该方法减少了标签闪烁，也避免特效在相邻情绪之间频繁切换。

## 6. 七类视频特效设计

每种表情都有独立的配色、主贴纸、局部元素和全屏粒子氛围。

特效由多个渲染层组合完成：

### 6.1 通用面部特效层

`EmotionEffects` 负责围绕人脸框绘制基础局部效果，并对人脸框进行进一步平滑。该层根据当前表情切换颜色、线条和局部装饰。

### 6.2 宽屏背景与局部动画层

`WideScreenEmotionRenderer` 负责：

- 在整个摄像头画面中生成分散的动态小元素；
- 保持人脸中心区域相对清晰；
- 为 angry、disgust、fear、neutral 和 sad 增加额外的人脸局部动画；
- 根据帧编号更新位置、缩放、透明度和运动状态。

### 6.3 Happy 与 Surprise 专用渲染器

Happy 和 Surprise 使用独立渲染器：

```text
HappyEffectRenderer
SurpriseEffectRenderer
```

它们具有更完整的贴纸组合、进入动画、漂浮元素和专属运动模式，因此与其余五类分开管理。

### 6.4 素材组织

七类素材位于：

```text
expert/assets/effects/<emotion>/
```

每个情绪目录使用 `manifest.json` 描述素材，包括：

```text
main      主云朵或主表情贴纸
core      较大的核心装饰元素
fragment  小型粒子与碎片元素
```

运行时使用透明底 PNG 进行 alpha blending。通过 manifest 管理素材可以避免在代码中写死所有文件名，也方便后续替换或增加贴纸。

## 7. 手帐风侧栏 UI

最终界面右侧使用 `ScrapbookSidebar` 绘制手帐风信息面板。侧栏显示：

```text
当前表情标题
分类置信度
Tracking 状态
PRED 预测耗时
RTMPose 模型名称
RATE 实际显示帧率
Automatic / Preview 模式
Effects On / Off
快捷键提示
对应情绪的小云朵贴纸
```

侧栏采用透明撕纸边缘，并略微覆盖摄像头画面。右侧边界保持固定，因此纸张覆盖视频的一小部分不会改变整体窗口尺寸。

指标含义：

- `PRED`：一次完整表情预测耗时，包括 RTMPose、特征构建、Scaler 和分类器；
- `RATE`：主界面的实际显示 FPS；
- 摄像头读取、人脸检测、特效绘制和 UI 合成不计入 `PRED`。

## 8. V10 异步推理优化

早期版本在主线程中同步执行：

```python
prediction = recognizer.predict(face_roi)
```

当 RTMPose 推理需要二十到三十多毫秒时，摄像头读取、动画更新和 UI 绘制都会等待，造成动画间歇性卡顿。

V10 将推理移动到一条专用后台线程：

```text
Main thread
camera → face detection → submit latest face → render effects → render UI

Worker thread
RTMPose → feature extraction → scaler → classifier → publish result
```

主线程不等待推理完成，而是继续渲染下一帧，并使用最近一次已经完成的预测结果。

### 单槽位 latest-frame buffer

V10 没有使用无限增长的普通队列，而是只保留一个待处理人脸：

```text
new face crop replaces the older waiting crop
```

如果推理线程仍在忙，新提交的人脸会覆盖尚未开始处理的旧人脸，因此不会积累过时帧，也不会出现明显的表情延迟。

线程结构只使用：

- 一条主线程；
- 一条 RTMPose 推理线程；
- 一个模型实例；
- 一个最新人脸槽位；
- 一个最新结果槽位。

不使用多个线程同时调用同一个 RTMPose 模型，以避免 GPU 争用、显存增加和线程安全问题。

## 9. 动画卡顿与闪烁的处理

最终版本针对视觉问题进行了以下处理：

- 检测框平滑，减少贴纸位置抖动；
- 丢失人脸时短暂保持上一帧框；
- 五次预测窗口投票，减少标签闪烁；
- 切换情绪或预览模式时重置对应 renderer；
- 动画每帧继续更新，不因后台推理暂停；
- 单槽位异步推理避免处理过时的人脸帧；
- `PRED` 与 `RATE` 分开显示，分别反映模型速度和界面流畅度。

## 10. 键盘操作

```text
0     返回自动识别模式
1     预览 Angry
2     预览 Disgust
3     预览 Fear
4     预览 Happy
5     预览 Neutral
6     预览 Sad
7     预览 Surprise
E     开启或关闭视频特效
D     显示或隐藏调试信息
Q     退出程序
```

预览模式不会重新运行或修改分类器，只是临时将指定表情作为显示标签，便于检查七类特效和 UI。

## 11. 主要文件说明

| 文件 | 作用 |
|---|---|
| `realtime_demo_v10_threaded_scrapbook.py` | 最终 V10 手帐界面入口 |
| `realtime_demo_v10_threaded.py` | 摄像头主循环、检测、平滑、动画和交互逻辑 |
| `async_expression_worker.py` | 单推理线程与 latest-frame buffer |
| `rtmpose_emotion_recognizer.py` | RTMPose 关键点、318 维特征与分类器接入 |
| `robust_face_detector.py` | MediaPipe 主检测、Haar fallback、平滑与短时保持 |
| `emotion_effects.py` | 通用人脸局部特效 |
| `wide_screen_emotion_renderer_v8f.py` | 全屏粒子和五类局部动画 |
| `happy_effect_renderer_v8f.py` | Happy 专属动画 |
| `surprise_effect_renderer_v8f.py` | Surprise 专属动画 |
| `scrapbook_sidebar.py` | 手帐风右侧信息面板 |
| `assets/effects/` | 七类透明 PNG 动画素材和 manifest |
| `assets/ui/` | 手帐侧栏标题、纸张、胶带、底条和云朵素材 |
| `models/rtmpose_expression/` | 已训练分类器与 StandardScaler |

## 12. 最终特点

- 使用关键点而不是整张人脸图像进行表情分类；
- 支持七种 FER 表情；
- 使用 MediaPipe + Haar fallback 提高人脸检测鲁棒性；
- 使用时间投票降低标签闪烁；
- 使用透明贴纸和多层粒子完成动态视频特效；
- 使用手帐风侧栏统一展示分类和性能信息；
- 使用单 RTMPose 后台线程减少主界面阻塞；
- 保留实时预览、调试和特效开关，便于演示与测试。
