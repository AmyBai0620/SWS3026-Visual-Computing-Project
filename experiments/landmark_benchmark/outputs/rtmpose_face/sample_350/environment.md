# RTMPose-Face benchmark environment

- Python: 3.10.20
- Platform: Windows 11
- NumPy: 1.26.4
- OpenCV: 4.11.0
- PyTorch: 2.1.2+cu118
- Torch CUDA build: 11.8
- CUDA available: True
- Benchmark device: cpu
- Torch CPU threads: 16
- MMCV: 2.1.0
- MMEngine: 0.10.7
- MMDetection: 3.2.0
- MMPose: 1.3.2
- Model: RTMPose-M Face6 256x256
- Config: MMPose built-in `rtmpose-m_8xb256-120e_face6-256x256.py`
- Checkpoint: `rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth`
- Score threshold: 0.2
- Warm-up runs: 5

The installed PyTorch/MMCV build supports CUDA, but the benchmark was explicitly run with `device="cpu"`.
