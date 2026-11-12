# 逐帧深度：第一帧验证（2026-11-11）

## 本次完成范围与复现

已实现 Depth Anything V2-Small 逐帧脚本，并在指定视频第 0 帧上完成 GPU
推理。它不使用帧间信息、光流、尺度拟合或米制转换。本次是首帧验证，
不宣称完成 100 帧实验或时间稳定性评价。

在项目根目录，激活环境后运行：

```powershell
conda activate cv_project
python -m src.monocular.run_framewise --run-name first_frame_check --max-frames 1
python scripts/verify_framewise_run.py --run-dir outputs/framewise/first_frame_check
```

已交付结果为 `outputs/framewise/first_frame/`；重跑请用新名称。
配置为 `configs/framewise.yaml`，依赖版本为 `requirements-framewise.txt`。

## 实测结果

| 项目 | 结果 |
|---|---|
| 模型 | Depth Anything V2-Small / vits，24,785,089 个参数 |
| 设备与精度 | RTX 4070 Laptop，FP32，关闭 TF32 |
| 原图 | 第 0 帧，960×540，0 秒；源帧率 24 FPS |
| 网络输入 | 1×3×518×924 |
| 原始预测 | float32，540×960，100% 有限值 |
| 数值范围 | 0 到 3.042626858，相对输出，不是米 |
| 预热 | 1 次，不计入测量 |
| 单帧 forward 加输出缩放 | 66.07 ms |
| 加上预处理与传输 | 90.94 ms |
| PyTorch 峰值 allocated | 413.57 MiB |
| PyTorch 峰值 reserved | 574.00 MiB |

一次测量不是完整视频的稳定平均速度。显存统计包含模型权重，来自
PyTorch 分配器，不代表显卡或进程的全部显存。官方打印 xFormers not
available 时自动使用内置注意力；本次已验证这条路径，无需安装 xFormers。

## 正确性核验

使用官方 Resize、NormalizeImage、PrepareForNet，保持 BGR→RGB/255、
lower_bound 等比例缩放、14 的倍数、OpenCV 三次插值、ImageNet 归一化及
CHW float32。输出按官方 bilinear / align_corners=True 恢复原尺寸。
输入张量与官方 image2tensor 逐元素一致；预测与官方 infer_image 的最大
绝对差为 0。视频、权重、源码和本次推理代码 SHA256 均已记录并核对。
保存的 PNG 与解码首帧一致，色图能从保存的浮点预测和固定范围精确重建。

## 文件

- `src/monocular/engine.py`：官方模型加载、预处理、独立推理与计时。
- `src/monocular/run_framewise.py`：前缀检查、顺序推理、浮点保存和运行记录。
- `src/monocular/visualization.py`：整段固定范围上色与 RGB/深度拼接。
- `scripts/verify_framewise_run.py`：原图、官方接口、哈希与上色核验。
- `outputs/framewise/first_frame/frames/000000/relative_depth.npy`：原始相对深度。
- `results/figures/framewise/first_frame_preview.png`：精选并排预览。
- `results/tables/framewise/first_frame_*.json`：统计、来源和验证记录。

先保存所有请求帧的浮点预测，再确定整段全局 min/max，避免逐帧归一化。
首帧范围只用于本次预览；完整片段需要对全片段重新确定范围。
单帧仅生成 PNG。后续已完成现有 70 帧的多帧推理和 MP4 输出，详见
`src/monocular/README.md` 和 `outputs/framewise/rollercoaster_70/`；上述表格仍是首帧历史记录。

## 指定视频的 70 帧问题

完整解码检查了 `data/videos/davis_rollercoaster.mp4` 与官方代码目录中的
`external/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4`。
两者均为 1,809,560 字节、960×540、24 FPS、实际 70 帧，SHA256 相同：

```text
7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8
```

本地 Video Depth Anything 提交为
`4f5ae23172ba60fd7bc11ef671cca678842c7072`。视频 Git blob 为
`f385e1bb30f90b48e0c6a8f485b8ff4862f082a5`，与提交中记录一致，
不是本地复制导致的截断。需核对课程是否另有视频版本，或确认不足 100 帧
时的处理口径。首帧任务不受影响。

请求 `--max-frames 100` 时，脚本会在预先解码到 70 帧后报错，且不会加载
模型或创建结果目录。不会通过重复、补帧、重采样或静默截断来满足数量。
