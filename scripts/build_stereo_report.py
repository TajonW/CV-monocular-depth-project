"""Build the Chinese stereo mini-report PDF and editable Markdown from saved results.

Requires reportlab. Run summarize_stereo.py in the experiment environment first.
Windows uses SimHei; on other systems pass --font a Chinese-capable TTF file.
"""

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", default=r"C:\Windows\Fonts\simhei.ttf")
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont("Chinese", args.font))
    tables = ROOT / "results/tables/stereo"
    figures = ROOT / "results/figures/stereo"
    report_dir = ROOT / "report/stereo"
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline = list(csv.DictReader((tables / "baseline_metrics.csv").open(encoding="utf-8")))
    summaries = json.loads((tables / "summary.json").read_text())
    summary = next(s for s in summaries if s["run"] == "baseline")
    manifest = json.loads((tables / "baseline_manifest.json").read_text())
    lr_path = tables / "left_right_summary.json"
    lr = json.loads(lr_path.read_text()) if lr_path.exists() else None
    style = ParagraphStyle("body", fontName="Chinese", fontSize=10, leading=16,
                           textColor=colors.HexColor("#263445"), wordWrap="CJK", spaceAfter=8)
    heading = ParagraphStyle("heading", parent=style, fontSize=17, leading=23, spaceAfter=12,
                             textColor=colors.HexColor("#153d59"))
    title_style = ParagraphStyle("title", parent=heading, fontSize=25, leading=34, spaceAfter=16)
    small = ParagraphStyle("small", parent=style, fontSize=8.5, leading=13)
    code_style = ParagraphStyle("code", parent=style, fontName="Courier", fontSize=8.5, leading=13,
                                backColor=colors.HexColor("#f1f5f8"), borderPadding=6)
    story, markdown = [], []
    width = A4[0] - 88

    def text(value, kind=style):
        value = value.replace("−", "-").replace("–", "-").replace("²", "^2")
        story.append(Paragraph(escape(value), kind))
        markdown.append(value + "\n")

    def heading_text(value):
        story.append(Paragraph(escape(value), heading))
        markdown.append("## " + value + "\n")

    def table(data, widths=None):
        obj = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        obj.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Chinese"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153d59")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f2f6f9"), colors.white]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story.extend([obj, Spacer(1, 10)])
        markdown.append("| " + " | ".join(map(str, data[0])) + " |\n|" + "---|"*len(data[0]))
        markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in data[1:])
        markdown.append("")

    def image(name, height):
        story.append(Image(str(figures / name), width=width, height=height, kind="proportional"))
        story.append(Spacer(1, 9))
        markdown.append(f"![{name}](../../results/figures/stereo/{name})\n")

    def command(value):
        story.append(Paragraph(escape(value).replace("\n", "<br/>"), code_style))
        story.append(Spacer(1, 6))
        markdown.append("```powershell\n" + value + "\n```\n")

    def page():
        story.append(PageBreak())

    text("Part 1 · 经典双目深度实验", title_style)
    text("Classical Stereo: Disparity to Depth", heading)
    text("AIAA 3201 | Tingzhi WANG · Dongyou LI | 基线：2026-11-06；左右诊断：2026-11-08", small)
    text("摘要：在 Middlebury 15 个四分之一分辨率场景上，使用固定参数的 OpenCV StereoSGBM 计算左视角视差，并结合各场景标定转换为米制深度。全程不训练模型，不使用 GPU。报告覆盖定量评价、窗口敏感性实验、可视化、三类困难案例及复现方法。")
    table([["场景平均 MAE", "场景平均有效率", "平均匹配耗时"],
           [f"{summary['macro_mae_px']:.3f} px", f"{summary['macro_coverage']:.2%}", f"{summary['macro_matcher_ms']:.2f} ms / pair"]], [width/3]*3)
    heading_text("01  数据、方法与单位")
    text("数据：使用 trainingQ 的全部 15 对原始尺寸图像。输入为 im0.png、im1.png 和 calib.txt；disp0GT.pfm 仅用于评估，不参与匹配。未使用 testQ、NYU 或神经网络模型。")
    text("流程：左右 RGB → 灰度图 → StereoSGBM → 浮点像素视差与无效掩码 → 标定转换 → 米制深度；原始浮点预测单独保存。")
    command("d_px = raw_disparity / 16.0\nZ_m = f_px * baseline_mm / (d_px + doffs_px) / 1000.0")
    text("标定：保持四分之一分辨率，不再次缩放焦距；doffs = cx_right - cx_left。深度指沿光轴方向的 Z，而非到光心的欧氏距离。Adirondack 的 f=1038.018 px、B=176.252 mm、doffs=53.271 px；假设 d=40 px，可得 Z≈1.962 m。")
    text("PFM：按头部符号确定字节序，翻转底部起始的行顺序，直接读取像素视差。该数据的头部比例绝对值不用于再次缩放视差。无效预测保存为 NaN，绝不补成零深度。")
    heading_text("固定基线参数")
    text("灰度输入，MODE_SGBM；minDisparity=0；numDisparities 为 calib.txt 中 ndisp 向上取整到 16 的倍数；blockSize=5；P1=8×5²=200，P2=32×5²=800；uniquenessRatio=10；speckleWindowSize=100；speckleRange=2；disp12MaxDiff=1；preFilterCap=31。未进行逐场景调参、孔洞填补或额外后处理。")

    page()
    heading_text("02  定量结果与窗口敏感性")
    text("真值有效域 G：视差有限且大于 0。预测有效域 P：OpenCV 原始值大于 (minDisparity−1)×16，并且解码后有限。MAE 在 G∩P 上计算；有效率 = |G∩P| / |G|。无效像素不计入 MAE，但通过有效率暴露拒绝预测的比例。")
    data = [["场景", "尺寸 W×H", "MAE / px", "有效率", "匹配 / ms"]]
    for row in baseline:
        data.append([row["scene"], f"{row['width']}×{row['height']}", f"{float(row['mae_px']):.3f}",
                     f"{float(row['coverage']):.2%}", f"{float(row['matcher_ms']):.2f}"])
    data.append(["场景等权平均", "15 scenes", f"{summary['macro_mae_px']:.3f}", f"{summary['macro_coverage']:.2%}", f"{summary['macro_matcher_ms']:.2f}"])
    table(data, [134, 104, 80, 92, width-410])
    text(f"汇总采用每场景等权平均；另存的像素加权结果为 MAE={summary['pooled_mae_px']:.3f} px、有效率={summary['pooled_coverage']:.2%}。这是课程协议下的自测结果，不代表 Middlebury 官方排行榜成绩。", small)
    table([["窗口", "平均 MAE / px", "平均有效率", "平均匹配 / ms"]] +
          [[str(s["block_size"]), f"{s['macro_mae_px']:.3f}", f"{s['macro_coverage']:.2%}", f"{s['macro_matcher_ms']:.2f}"] for s in summaries], [65, 145, 145, width-355])
    text("敏感性实验保持数据、掩码与其他配置一致；改变窗口时按相同公式更新 P1/P2，因此检验的是窗口与对应平滑惩罚的联合设置。窗口 3 的误差最低但覆盖率略低，窗口 9 更容易跨越深度边界。固定窗口 5 作为原始基线，不依据真值挑选替换。该实验不是 Part 3 改进。", small)

    page()
    heading_text("03  视差、深度与场景结构")
    image("qualitative.png", 370)
    text("图 1：从上到下为 Adirondack、Teddy、Jadeplant；列依次为左 RGB、真实视差、预测视差和预测米制深度。灰色为无效区域；每行真实与预测视差共用色阶。深度显示上限为该场景有效预测的第 99 百分位，原始数值不裁剪。", small)
    text("Adirondack：主要平面和近远层次得到恢复，但左边界和细小结构附近仍有缺失。图像左侧较宽无效带部分来自视差搜索的可匹配范围限制，不应误认为真实零深度。")
    text("Teddy：在此次基线中误差最低（0.714 px）；主体结构较清晰。低误差仍伴随未预测像素，因此必须同时查看有效率和无效掩码。")
    text("Jadeplant：枝叶交错、细结构和背景交界处存在大量拒绝匹配；有效率仅 55.57%，剩余有效预测的 MAE 为 5.308 px。单看填色后的深度图会掩盖这些缺失。")
    text("所有 15 个场景的完整八联图位于 outputs/stereo/baseline/<scene>/overview.png，包含左右图、真实/预测视差、深度、预测无效掩码、绝对误差及真值有效掩码。")

    page()
    heading_text("04  三个实际困难案例与局限")
    image("failure_cases.png", 360)
    text("图 2：固定局部裁剪；各行为 PianoL、Jadeplant、Pipes；各列为左图、右图、预测视差、绝对视差误差（0–10 px 显示）。两幅 RGB 使用相同图像坐标裁剪，并非已按对应点对齐。灰色表示无效/不可评估。裁剪坐标保存在 failure_rois.json。", small)
    text("① PianoL：灯具附近左右图照明差异明显，墙面同时缺少纹理；预测出现大片错误或空洞。全场 MAE=6.186 px，为基线最差场景。亮度变化与低纹理可能共同导致匹配代价不可靠；这里是观察解释，未做因果分离实验。")
    text("② Jadeplant：细枝、叶片和遮挡边界附近大量像素无效，局部错误延伸到背景。小结构与块匹配窗口混合前景/背景，是可见的困难因素。")
    text("③ Pipes：管道交叉、重复形状和深度不连续处误差集中，边缘出现空洞和错误延伸。场景平均 MAE=2.266 px；平均指标不能替代局部诊断。")
    text("局限：只评估 15 个给定训练场景；没有独立测试集泛化结论。新增左右一致性诊断见后页；尚未实现学习式置信度模型或深度修补。主要误差指标是视差误差，未将其等同于米制深度精度；误差传播随场景标定和距离而改变。")

    if lr is not None:
        page()
        heading_text("05  独立左右一致性诊断")
        text("保持原基线预测不变，交换左右图并重新运行 StereoSGBM 得到带符号右视差。若左视差定义为 dL=xL-xR，右视差为 dR=xR-xL，则右图对应位置为 xR=xL-dL；以最近像素采样右视差，检查 |dL+dR(xR,y)|≤1 px。像素映射不加 doffs。", small)
        text("右搜索范围为 [-minL-N+1,-minL]；两侧均沿用原内部筛选参数。越界或任一侧无效时标为不可检查，不能当作一致。右视差已除以 16，按负 minDisparity 单独解码其无效标记。", small)
        table([["场景等权平均", "原始基线", "仅保留 LR 通过点"],
               ["视差 MAE / px", f"{lr['baseline_mae_px']:.3f}", f"{lr['lr_mae_px']:.3f}"],
               ["有效预测比例", f"{lr['baseline_coverage']:.2%}", f"{lr['lr_coverage']:.2%}"]], [width/3]*3)
        image("left_right_diagnostic.png", 340)
        text("图 3：Adirondack、Jadeplant、PianoL 的 RGB、五状态诊断图和左视角残差（显示截断至 5 px）。灰：左无效；蓝：映射越界；黄：右无效；红：双向残差超阈值；绿：通过。灰色残差为不可检查，不能视为零误差。", small)
        text(f"可检查点的场景平均通过率为 {lr['pass_fraction_checkable']:.2%}。筛选后 MAE 下降，但评价像素集合缩小，有效率下降 {(lr['baseline_coverage']-lr['lr_coverage'])*100:.2f} 个百分点；因此只是可靠性与覆盖率的取舍，不代表完整深度图得到同等幅度提升。一致也不保证真值正确，不一致也不等于真实遮挡。", small)
        text(f"额外右匹配平均耗时 {lr['right_matcher_ms']:.2f} ms/对，向量化检查约 {lr['check_ms']:.2f} ms/对；这与前面的基线时间分别测量，未当作同一端到端基准。原始浮点诊断、掩码、逐场景统计与来源哈希保存在 outputs/stereo/left_right/。", small)

    page()
    heading_text("06  复现、验证与交付索引" if lr is not None else "05  复现、验证与交付索引")
    text(f"运行平台：{manifest['cpu']}；Windows；Python 3.11.18；CPU 单 OpenCV 线程，未使用 GPU，实验 GPU 显存占用为 0。OpenCV={manifest['packages']['opencv-python']}，NumPy={manifest['packages']['numpy']}，Matplotlib={manifest['packages']['matplotlib']}，PyYAML={manifest['packages']['PyYAML']}。")
    text("时间口径：每场景先预热 1 次，再测量 3 次 matcher.compute，取中位数；上表耗时再对 15 个场景等权平均。不包含读盘、灰度转换、评估、绘图和压缩保存；单次短基准会受系统负载影响。")
    text("在项目根目录、激活 Python 环境后执行；完成过的 run-name 不会被覆盖，重跑请指定新名称。", small)
    for cmd in ["python -m pip install -r requirements-stereo.txt",
                "python scripts/prepare_middlebury.py",
                "python -m unittest discover -s tests -v",
                "python -m src.stereo.run_stereo --run-name baseline",
                "python -m src.stereo.evaluate_stereo --run-dir outputs/stereo/baseline",
                "python -m src.stereo.run_stereo --run-name window3 --block-size 3",
                "python -m src.stereo.run_stereo --run-name window9 --block-size 9",
                "python scripts/summarize_stereo.py"] + (["python -m src.stereo.diagnose_left_right"] if lr is not None else []):
        command(cmd)
    text("验证：8 项单元测试覆盖 PFM 字节序/方向、头部比例、定点视差、深度单位、评价掩码、合成平移，以及左右检查的符号、边界、亚像素取样、零视差、阈值与无效状态。另重算 15 场景基线指标，核验 PFM 读取结果；新增诊断核对全部场景状态划分和基线文件哈希。", small)
    text("文件：源代码在 src/stereo/；配置在 configs/stereo.yaml；完整浮点预测、逐场景参数、输入/代码 SHA256 和环境快照在 outputs/stereo/<run>/；精选 CSV/JSON 在 results/tables/stereo/，图片在 results/figures/stereo/；本报告及可编辑 Markdown 在 report/stereo/。", small)
    text("PDF 重建：安装 requirements-report.txt 后运行 python scripts/build_stereo_report.py；默认使用 Windows 黑体，其他系统通过 --font 指定支持中文的 TTF。", small)
    text("参考资料：[1] Hirschmüller, Stereo Processing by Semiglobal Matching and Mutual Information, TPAMI 2008. [2] Middlebury Stereo v3 / 2014 数据与标定说明. [3] OpenCV StereoSGBM 文档. [4] 本课程 Project 3 说明，第 5.1、6.1、6.2 节。", small)
    for url in ["https://elib.dlr.de/55367/", "https://vision.middlebury.edu/stereo/data/scenes2014/", "https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html"]:
        text(url, small)
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e1e9"))
        canvas.line(44, 38, A4[0]-44, 38)
        canvas.setFont("Chinese", 8)
        canvas.setFillColor(colors.HexColor("#667788"))
        canvas.drawString(44, 24, "AIAA 3201 | Classical Stereo | updated 2026-11-08")
        canvas.drawRightString(A4[0]-44, 24, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(report_dir / "part1_stereo_report.pdf"), pagesize=A4,
                            leftMargin=44, rightMargin=44, topMargin=40, bottomMargin=48,
                            title="Part 1 Classical Stereo: Disparity to Depth", author="Tingzhi WANG; Dongyou LI")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (report_dir / "part1_stereo_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(report_dir / "part1_stereo_report.pdf")


if __name__ == "__main__":
    main()
