# 左右一致性诊断（2026-11-08）

## 目的与运行

检查左图预测的对应点，在右图独立预测中能否返回原位置。
该诊断补充课程的 optional diagnostic，原始基线及三个窗口实验不变。
不需要 opencv-contrib、训练模型、GPU 或新增 Python 依赖。

在项目根目录的已激活环境中运行：

```powershell
python -m src.stereo.diagnose_left_right
```

默认从 `outputs/stereo/baseline/` 读取，写入 `outputs/stereo/left_right/`。
已存在的非空诊断目录不会被覆盖；复现时可指定其他目录：

```powershell
python -m src.stereo.diagnose_left_right --output-dir outputs/stereo/left_right_repro
```

## 坐标、符号与阈值

左右图已经校正，匹配点使用相同的行坐标 y。

```text
dL(xL,y) = xL - xR
dR(xR,y) = xR - xL
xR = xL - dL(xL,y)
residual = abs(dL(xL,y) + dR(xR,y))
consistent = checkable AND residual <= 1.0 pixel
```

右图作为 reference，交换图像顺序重新运行 StereoSGBM。
右视差为负值，不使用绝对值参与残差运算。假设左搜索范围
`[minL, minL+N-1]`，右搜索范围为 `[-minL-N+1, -minL]`。
例如 Adirondack 左范围为 `[0,79]`，右范围为 `[-79,0]`。
右原始定点结果也除以 16，其无效标记按右 minDisparity 单独判断。

右图浮点对应坐标使用最近像素 `floor(xR+0.5)`，先检查连续坐标
位于 `[0,W-1]` 内，禁止越界环绕或外推。当前检查不做双线性插值，
因此亚像素边界的结果受此离散采样约定影响。
`doffs` 只用于视差到深度转换，不添加到上述像素对应映射。

左右匹配器均保留原基线的内部 `disp12MaxDiff=1` 等设置。
因此这是在已有内部筛选结果上的额外双向诊断，不是关闭内部筛选后的原始匹配评估。
阈值固定为 1 px，未使用真值逐场景选择阈值。

## 五种互斥状态

| 编码 | 图中颜色 | 定义 |
|---|---|---|
| 0 | 灰 | 左预测无效，无法发起检查 |
| 1 | 蓝 | 左预测有效，但映射到右图后越界 |
| 2 | 黄 | 映射在图内，但取样处右预测无效 |
| 3 | 红 | 双方有效且可检查，但残差大于阈值 |
| 4 | 绿 | 可检查且残差小于等于阈值 |

0/1/2 不等同于 residual=0，也不属于“通过”。保存的 residual_px 在这些位置为 NaN。
status.png 保存 0-4 整数标签；overview.png 中提供彩色图例。
consistent_mask.png、inconsistent_mask.png 为 0/255 二值图。
右视差幅值图位于右视角，残差与状态图位于左视角，不可直接按同一坐标当成同一物点。

## 实际结果与解释

15 个场景等权平均：

| 评价 | 原基线 | 仅保留一致点 |
|---|---:|---:|
| 视差 MAE | 2.033342 px | 1.513458 px |
| 有效预测比例（相对有限且正的真值） | 75.864931% | 63.719293% |

可检查点上的平均通过率为 96.681630%；该分母只包括状态 3/4，
不包含右无效等不可检查点，也不使用真值掩码，因此它不是深度预测准确率。

筛选后评价集合缩小，有效率下降约 12.15 个百分点。
降低 MAE 是拒绝部分预测后的条件误差变化，不表示全图误差改善。
左右一致不能保证预测符合真值；不一致也不等于真实遮挡标签。
若要讨论遮挡，需要额外真值或专门的遮挡分析，不能直接将红色区域命名为遮挡。

## 输出与复现记录

完整输出为每场景的 diagnostic.npz、overview.png、三种标签/二值图、
metrics.json 和 right_parameters.json，以及整体 metrics.csv、summary.json、
manifest.json。manifest 记录来源基线预测 SHA256、输入核对、阈值和诊断代码哈希。
右匹配的额外耗时按原有预热/重复方式单独测量；检查时间单独记录，
不可与先前不同时间测量的左匹配直接合并后宣称端到端速度。

精选图在 `results/figures/stereo/*left_right*.png`，统计和来源记录在
`results/tables/stereo/left_right_*`。报告生成器会自动读取这些文件，
加入左右一致性页面；基线 CSV、浮点预测和基线统计均不修改。
