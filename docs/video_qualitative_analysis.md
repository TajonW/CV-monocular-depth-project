# Part 2：视频深度定性分析

## 结论

在现有 70 帧片段中，两种 Small 模型均保留了主要轨道、立柱和车辆的场景结构。
VDA 在片尾树冠这一局部区域表现出更稳定的相对输出；但细杆、车辆细节及远景
植被仍有共同不足。现有证据支持局部显示稳定性改善，不支持“VDA 在所有区域
全面更好”“消除了运动伪影”或“米制深度更准确”的结论。

## 设置与观察方法

- 源视频：davis_rollercoaster.mp4，真实帧 0–69，24 FPS，2.916667 秒。
- DA V2-Small：逐帧独立推理，FP32，网络空间输入 924×518，原生深度 960×540。
- VDA-Small：官方离线推理，FP16 autocast，网络空间输入 924×518，原生深度 640×360。
- 显示统一为 640×360，采用相同 inferno 颜色索引；各方法沿用整段固定范围：
  DA [0,3.818237782]，VDA [0,7.433829784]。两者颜色不是共同的物理距离。
- 分析依据为全片分段概览和带帧号的连续帧放大图，另有树冠 RGB 特征跟踪诊断。
  没有修改浮点预测、重新推理、增加时序平滑或拟合跨模型尺度。
- 原始对比视频维持 24 FPS。辅助慢放副本为 6 FPS、约 11.67 秒，仅供观察；
  副本仍只有 70 个画面，保留原始帧号和源时间，不是补帧至 100 帧。

## 1. 时间稳定性：片尾树冠的局部改善

证据：`results/figures/video_analysis/tree_64_69.png` 和
`tree_track_diagnostic.png`；帧 64–69，源时间 2.667–2.875 秒。

固定屏幕坐标相邻帧的差异会混入相机平移、旋转、物体进出和遮挡，不能直接当作
闪烁误差。为辅助观察，在 RGB 第 64 帧的树冠内部区域 x=[555,632)、y=[315,353)
检测角点，用 pyramidal Lucas–Kanade 跟踪，并检查前后向位置误差 <1 像素、LK
局部误差 <20。18 条候选轨迹均通过。没有用深度输出选择轨迹；主图显示按 RGB
角点强度排序的首条有效轨迹。

主轨迹从 (585,347) 移到约 (567.20,306.21)，最大前后向位置误差约 0.0050 px。
在各自原有整段色阶下，该点的归一化显示值为：

| 原始帧号 | DA V2-Small | VDA-Small |
|---|---:|---:|
| 64 | 0.02627 | 0.05895 |
| 65 | 0.03688 | 0.06224 |
| 66 | 0.04462 | 0.06279 |
| 67 | 0.06103 | 0.05899 |
| 68 | 0.05506 | 0.05587 |
| 69 | 0.04791 | 0.05724 |

该点在 VDA 下变化更小，连续帧图也显示树冠内部颜色较平稳。同一区域 18 条轨迹
中有 17 条呈现较小的 VDA 显示值极差，另 1 条相反；所有轨迹均保存在 CSV，
没有只保留支持结论的点。

这是局部的显示稳定性证据，不是全片时间误差或物理精度分数。18 个点来自同一
小区域、仅六帧，彼此有相关性；即便跟踪正确，相机运动也可能改变真实深度。
方法本身又具有不同的任意相对尺度，不能把这些变化换算成米或宣称全局闪烁
降低某个百分比。模型和设置也不构成时序模块的单独消融。

## 2. 边缘稳定性：主要结构连续，细节没有一致优势

证据：`fine_support_00_05.png`（0–5 帧，0–0.208 秒）及
`junction_54_59.png`（54–59 帧，2.250–2.458 秒）。

两种方法都保持了支架大轮廓及连接结构。与 RGB 对照，较细的杆件、轨道孔隙和
交叉处轮廓出现增厚、局部间隙变小或细节粗化。VDA 的局部外观与 DA 接近，
没有足够清晰的证据支持其在这些区域持续优于 DA。

这些是当前共同显示尺寸下的轮廓观察。两种方法的原生输出尺寸和推理精度不同，
输出插值与显示分辨率也可能影响边缘表现，不能把所有边缘差异都归因于时序建模。

## 3. 运动与遮挡：能跟随车辆，但细部仍粗

证据：`vehicle_34_39.png`，34–39 帧，1.417–1.625 秒。

两种方法的车辆轮廓都随 RGB 中的位置变化而移动，保持了车体、轨道和附近支架的
主要形态。乘客、座椅及车辆的小轮廓在深度图中较粗，部分细小空隙没有保留下来。
该选段没有足够明确的证据证明 VDA 显著减少了长尾拖影或重影，因此不作此结论。

车辆与轨道颜色接近本身不是错误：它们可能确实深度相近，深度估计也不是语义
分割。这里只讨论相对于 RGB 的可见轮廓细节及帧间外观，不假定相邻物体必须异色。

## 4. 远景不足：树并未消失，但仍缺少细节和对比度

证据：`tree_64_69.png`。两种结果都存在深紫色的树冠轮廓，与近黑色天空有区分，
因此不能说“树完全没有预测”。树冠处于整段相对输出范围的低端，视觉对比度较低；
细叶片、枝条和叶间空隙远少于 RGB 中的细节。

VDA 在该局部更平稳，并不等于恢复了这些空间细节。不要通过逐帧增强或逐帧归一化
把这一不足隐藏掉；当前证据图始终使用原有整段固定色阶。

## 当前运行成本

| 方法 | 精度 | 推理流程 ms/真实输出帧 | 预热次数 | 峰值 allocated MiB | 峰值 reserved MiB |
|---|---|---:|---:|---:|---:|
| DA V2-Small | FP32 | 79.03 | 1 | 413.57 | 574 |
| VDA-Small | FP16 autocast | 790.09 | 0 | 8014.22 | 11858 |

这是已完成运行的记录，不是统一预热后的架构速度榜。DA 的流程包括独立预处理、
前向、输出缩放及回传；VDA 还包括重叠窗口计算和跨窗口对齐。两者均不含文件
解码和写出，但输出尺寸、精度、预热及注意力路径不同。不能据此声称 VDA 普遍
慢十倍。VDA 本次峰值分配接近本机约 8 GiB 容量，但运行没有 OOM；PyTorch
reserved 是分配器统计，不等于独占物理显存使用量。

## 可用于报告的英文段落

On the available 70-frame clip, both Small models preserve the principal track
and support structures. Case-based inspection reveals shared limitations in
thin members, small vehicle details and low-contrast distant vegetation. In a
tree-region example over frames 64–69, RGB-tracked features show smaller
variation in VDA's fixed-range display values than in the frame-wise baseline,
supporting a local temporal-stability benefit. However, this observation is
limited to one short region and does not establish global temporal-error
reduction or metric-depth accuracy. The selected moving-vehicle and support
sequences do not provide clear evidence of consistently superior edge detail
or substantially reduced ghosting. Our comparison therefore indicates a local
stability benefit while retaining important spatial-detail limitations. Both
methods use fixed clip-wide normalization; their colors do not share a metric
scale. Different native output sizes, precision settings and preprocessing
also limit causal attribution to temporal modeling alone.

## 复现与材料位置

```powershell
python scripts/analyze_video.py
```

- 主原速视频：`outputs/comparison/rollercoaster_70/rgb_da2_vda_comparison.mp4`。
- 辅助四倍慢放：`outputs/analysis/video_qualitative/comparison_slow_4x.mp4`。
- 放大图和概览：`results/figures/video_analysis/`。
- ROI、轨迹 CSV、诊断和溯源：`results/tables/video_analysis/`。
- 输出目录：`outputs/analysis/video_qualitative/`。

脚本重新检查原始对比视频所引用输入的哈希，核验慢放帧数/帧率，记录分析脚本
和库版本。观察结论由证据图支持，不由文件核验自动得出。
当前素材实际仅 70 帧；课程要求 100 帧的口径仍需确认。
