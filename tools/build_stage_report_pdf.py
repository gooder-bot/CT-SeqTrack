#!/usr/bin/env python3
"""Build the CT-SeqTrack staged progress report as a polished Chinese PDF.

The report is intentionally evidence-led. It distinguishes:
1. engineering completion;
2. descriptive tracking improvements;
3. matched causal-time controls;
4. candidate methods that are implemented but not yet validated.

Outputs:
  output/pdf/CT-SeqTrack_阶段性进展与成果汇报_20260724.pdf
  output/pdf/CT-SeqTrack_阶段性进展与成果汇报_20260724_source_manifest.json
  tmp/pdfs/ct_seqtrack_stage_report/qa/page_*.png
  tmp/pdfs/ct_seqtrack_stage_report/qa/contact_sheet_*.png
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_DATE = "2026-07-24"
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = WORKSPACE / "tmp" / "pdfs" / "ct_seqtrack_stage_report"
CHART_DIR = TMP_DIR / "charts"
QA_DIR = TMP_DIR / "qa"

PDF_PATH = OUTPUT_DIR / "CT-SeqTrack_阶段性进展与成果汇报_20260724.pdf"
MANIFEST_PATH = (
    OUTPUT_DIR
    / "CT-SeqTrack_阶段性进展与成果汇报_20260724_source_manifest.json"
)

FONT_REGULAR = Path(r"C:\Windows\Fonts\Deng.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\Dengb.ttf")
FONT_CHART = Path(r"C:\Windows\Fonts\msyh.ttc")

BLUE = colors.HexColor("#173B57")
BLUE_2 = colors.HexColor("#2F6B94")
BLUE_LIGHT = colors.HexColor("#EAF2F7")
GOLD = colors.HexColor("#C79A3B")
GOLD_LIGHT = colors.HexColor("#F7F0DF")
ORANGE = colors.HexColor("#D46A3A")
ORANGE_LIGHT = colors.HexColor("#FBEDE6")
INK = colors.HexColor("#1D2730")
SLATE = colors.HexColor("#5D6A76")
MID = colors.HexColor("#93A1AD")
LIGHT = colors.HexColor("#EEF2F4")
WHITE = colors.white


SOURCES = {
    "S1": {
        "title": "项目入口与当前状态",
        "path": "README.md",
        "role": "当前定位、阶段状态、核心结果与论文边界",
    },
    "S2": {
        "title": "研究计划与论文定位",
        "path": "refined_plan.md",
        "role": "创新点演化、相关工作边界、Go/No-Go 逻辑",
    },
    "S3": {
        "title": "已完成记录",
        "path": "done.md",
        "role": "从 2026-05-26 到 2026-07-24 的工程与实验时间线",
    },
    "S4": {
        "title": "实验结果汇总",
        "path": "sum_results.md",
        "role": "各轮消融结果、能说与不能说的边界",
    },
    "S5": {
        "title": "A1 时间编码对比",
        "path": "compare_results/reports/a1_time_encoding_comparison.md",
        "role": "raw、pseudo、MLP、Fourier 时间编码结果",
    },
    "S6": {
        "title": "五次稳定性复核",
        "path": "compare_results/reports/latest_5runs_comparison.md",
        "role": "A2 seed43/44、Gate 复测与稳定性风险",
    },
    "S7": {
        "title": "HTV 六组对比",
        "path": "compare_results/reports/htv_6runs_comparison.md",
        "role": "gap1124、burst-drop、random20 协议结果",
    },
    "S8": {
        "title": "Crop reachability 诊断",
        "path": "compare_results/reports/p0_ab_diagnostics_20260717.md",
        "role": "base、expanded、GT-history CV 的搜索可达性",
    },
    "S9": {
        "title": "递归历史可达性",
        "path": "compare_results/reports/p0b2_recursive_crop_reachability_20260717.md",
        "role": "previous-A1 与 predicted-history CV 的在线可行性边界",
    },
    "S10": {
        "title": "Reliability 开发集验证",
        "path": "compare_results/reports/p0b3_reliability_validation_20260720.md",
        "role": "observation-quality proxy、raw-CV 互补性和 selector",
    },
    "S11": {
        "title": "Reliability 独立冻结验证",
        "path": "compare_results/reports/p0b4_observation_reliability_validation_20260720.md",
        "role": "P0-B4 No-Go 证据",
    },
    "S12": {
        "title": "P0-C 时间负对照",
        "path": "compare_results/reports/p0c_frozen_protocol_validation_20260720.md",
        "role": "旧 feature-concat A2 的 true/fixed/shuffled 结果",
    },
    "S13": {
        "title": "TWC A/B/C 同提交对照",
        "path": "compare_results/reports/twc_abc_seed42_comparison_20260721.md",
        "role": "paired-view、TWC 净效应和端到端效应分解",
    },
    "S14": {
        "title": "M0-3/M0-4 机制分析",
        "path": "compare_results/reports/m0_m03_m04_analysis_20260721.md",
        "role": "proposal oracle 与 candidate 伪动力学",
    },
    "S15": {
        "title": "M1/M2 工程门禁",
        "path": "compare_results/reports/m1_m2_e0_e5_validation_20260722.md",
        "role": "E0-E5 数值、回退、梯度和几何不变量",
    },
    "S16": {
        "title": "M2 三组训练复核",
        "path": "compare_results/reports/m2_three_run_analysis_20260723.md",
        "role": "R1/R2/R3 完整性、final 与 late mean",
    },
    "S17": {
        "title": "M2 standard/gap1124 正式控制",
        "path": "compare_results/reports/m2_standard_gap8_analysis_20260724.md",
        "role": "八组 endpoint、bootstrap、tracking signal 与 causal-time No-Go",
    },
    "S18": {
        "title": "M2 scratch true/shuffled 复核",
        "path": "compare_results/reports/m2_scratch_time_analysis_20260724.md",
        "role": "random20 部分正信号、gap1124 不稳定与辅助监督冲突",
    },
    "S19": {
        "title": "M3/M4 第一阶段工程说明",
        "path": "M3_M4_IMPLEMENTATION.md",
        "role": "非对称 path distillation 与 fixed filter/tube 原型边界",
    },
    "S20": {
        "title": "SeqTrack3D 本地论文",
        "path": "2024SeqTrack3D Exploring Sequence Information for.pdf",
        "role": "固定序列、多帧点云和历史框基线",
    },
    "S21": {
        "title": "TrajTrack 本地论文与实现审计",
        "path": "../trajtrack/papers/2509.11453-TrajTrack.pdf",
        "role": "历史框 trajectory proposal 与 GT-free 评测边界",
    },
}


CHART_MAP = [
    {
        "section": "从固定帧序列到真实时间输入",
        "question": "真实秒数直接替换主干时间 token 是否有效？",
        "type": "grouped bar",
        "takeaway": "pseudo/order 语义接近基线，raw/MLP/Fourier 均明显退化。",
        "source": "S5",
    },
    {
        "section": "Dynamics 的早期正信号与稳定性风险",
        "question": "A2-order-dyn 的结果是否跨 seed 稳定？",
        "type": "grouped bar",
        "takeaway": "seed42 的 Precision 正信号没有在 seed43/44 稳定复现。",
        "source": "S6",
    },
    {
        "section": "TWC 从设想到 No-Go",
        "question": "paired-view 与一致性项分别贡献多少？",
        "type": "signed delta bars",
        "takeaway": "TWC 对 paired control 有正净效应，但端到端仍低于 single-view。",
        "source": "S13",
    },
    {
        "section": "HTV 协议揭示 protocol dependence",
        "question": "旧 A2 在不规则 cadence 下是否越困难越有效？",
        "type": "signed grouped bars",
        "takeaway": "random20 为正，gap1124 与 burst-drop 均显著为负。",
        "source": "S7",
    },
    {
        "section": "失败位置前移到 search crop",
        "question": "固定扩 crop 与轨迹 recenter 哪个更接近可达上限？",
        "type": "grouped bar",
        "takeaway": "GT-history CV 接近 99%，固定 2x crop 在强协议仍不足且背景成本高。",
        "source": "S8",
    },
    {
        "section": "M0 oracle 解锁 bounded innovation",
        "question": "dynamics proposal 是否具有 observation proposal 之外的信息？",
        "type": "bar",
        "takeaway": "d_dyn 明显降低离线误差，oracle 进一步证明插值空间存在。",
        "source": "S14",
    },
    {
        "section": "M2 standard 正信号",
        "question": "R1/R2/R3 的最终表现如何？",
        "type": "grouped bar",
        "takeaway": "R1 最强、R2 可训练，R3 shared-SE(2) W0 严重塌陷。",
        "source": "S16",
    },
    {
        "section": "正式时间负对照",
        "question": "M2 涨分是否由正确 physical time 导致？",
        "type": "signed delta bars",
        "takeaway": "M2-A1 为正，但 true-fixed/shuffled 均未形成因果优势。",
        "source": "S17",
    },
    {
        "section": "最新 scratch 训练",
        "question": "true-time 与 shuffled-time 分训是否支持 HTV 放大？",
        "type": "final-versus-late signed bars",
        "takeaway": "random20 稳定为正，gap1124 late window 反而为负。",
        "source": "S18",
    },
]


def register_fonts() -> None:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("DengXian fonts are required for Chinese PDF output.")
    pdfmetrics.registerFont(TTFont("Deng", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("Deng-Bold", str(FONT_BOLD)))
    pdfmetrics.registerFontFamily(
        "Deng",
        normal="Deng",
        bold="Deng-Bold",
        italic="Deng",
        boldItalic="Deng-Bold",
    )


def configure_matplotlib() -> font_manager.FontProperties:
    if not FONT_CHART.exists():
        raise FileNotFoundError("Microsoft YaHei is required for chart labels.")
    prop = font_manager.FontProperties(fname=str(FONT_CHART))
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    return prop


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def label_bars(ax, bars, fmt="{:+.2f}", offset=0.22, fontsize=9):
    for bar in bars:
        value = bar.get_height()
        if abs(value) < 1e-9:
            continue
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=fontsize,
            color="#25313A",
        )


def chart_seed_stability() -> Path:
    labels = ["seed42", "seed43", "seed44"]
    success = [50.96, 23.64, 46.90]
    precision = [63.31, 23.77, 52.62]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    bars1 = ax.bar(
        x - width / 2,
        success,
        width,
        label="Success",
        color="#2F6B94",
        edgecolor="#173B57",
    )
    bars2 = ax.bar(
        x + width / 2,
        precision,
        width,
        label="Precision",
        color="#D28A2E",
        edgecolor="#8D5B17",
    )
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_title("A2-order-dyn 跨 seed 最终指标", loc="left", fontweight="bold")
    ax.set_ylabel("分数")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 72)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(
        0.125,
        0.01,
        "nuScenes-mini, 60 epoch final；seed42 正信号未稳定复现。",
        fontsize=8.5,
        color="#5D6A76",
    )
    return save_figure(fig, CHART_DIR / "seed_stability.png")


def chart_crop_reachability() -> Path:
    protocols = ["standard", "gap1124", "burst-drop"]
    base = [85.41, 76.78, 77.72]
    expanded = [99.57, 89.08, 87.65]
    cv = [99.95, 98.96, 99.05]
    x = np.arange(len(protocols))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    bars1 = ax.bar(
        x - width,
        base,
        width,
        label="base crop",
        color="#AAB6BF",
        edgecolor="#6E7D88",
    )
    bars2 = ax.bar(
        x,
        expanded,
        width,
        label="2x expanded",
        color="#D28A2E",
        edgecolor="#8D5B17",
    )
    bars3 = ax.bar(
        x + width,
        cv,
        width,
        label="GT-history CV",
        color="#2F6B94",
        edgecolor="#173B57",
    )
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.45,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_title("不同搜索策略的目标点召回率", loc="left", fontweight="bold")
    ax.set_ylabel("mean target-point recall (%)")
    ax.set_xticks(x, protocols)
    ax.set_ylim(65, 103)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.text(
        0.125,
        0.01,
        "GT-history CV 是 oracle；2x crop 的平均点数约为 base 的 5.2-5.7 倍。",
        fontsize=8.5,
        color="#5D6A76",
    )
    return save_figure(fig, CHART_DIR / "crop_reachability.png")


def chart_m0_oracle() -> Path:
    labels = ["Observation d_obs", "Dynamics d_dyn", "Oracle blend"]
    values = [1.349, 0.309, 0.232]
    colors_ = ["#AAB6BF", "#2F6B94", "#D28A2E"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(
        labels,
        values,
        color=colors_,
        edgecolor=["#6E7D88", "#173B57", "#8D5B17"],
        width=0.58,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{bar.get_height():.3f} m",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("M0-3 crop-reachable cohort 的 proposal 误差", loc="left", fontweight="bold")
    ax.set_ylabel("中心误差（m，越低越好）")
    ax.set_ylim(0, 1.55)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.125,
        0.01,
        "1,311 endpoints / 213 tracklets；这是离线机制证据，不是在线 tracking 分数。",
        fontsize=8.5,
        color="#5D6A76",
    )
    return save_figure(fig, CHART_DIR / "m0_proposal_oracle.png")


def chart_formal_controls() -> Path:
    labels = [
        "standard\nM2-A1",
        "gap1124\nM2-A1",
        "standard\ntrue-fixed",
        "standard\ntrue-shuffled",
        "gap1124\ntrue-fixed",
        "gap1124\ntrue-shuffled",
    ]
    success = [4.133, 2.279, 0.031, 0.068, -0.127, -0.318]
    precision = [9.445, 4.143, -0.010, 0.085, 0.014, -0.209]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    b1 = ax.bar(
        x - width / 2,
        success,
        width,
        label="Success delta",
        color="#2F6B94",
        edgecolor="#173B57",
    )
    b2 = ax.bar(
        x + width / 2,
        precision,
        width,
        label="Precision delta",
        color="#D28A2E",
        edgecolor="#8D5B17",
    )
    ax.axhline(0, color="#25313A", linewidth=1.0)
    label_bars(ax, b1, offset=0.12, fontsize=8)
    label_bars(ax, b2, offset=0.12, fontsize=8)
    ax.set_title("R1 正式 matched comparison 与时间负对照", loc="left", fontweight="bold")
    ax.set_ylabel("差值（百分点）")
    ax.set_xticks(x, labels)
    ax.set_ylim(-1.2, 11)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(
        0.125,
        0.01,
        "M2-A1 的 tracklet bootstrap 95% CI 均为正；所有 time-control CI 均跨 0。",
        fontsize=8.5,
        color="#5D6A76",
    )
    return save_figure(fig, CHART_DIR / "formal_controls.png")


def chart_scratch_time() -> Path:
    labels = [
        "random20\nSuccess",
        "random20\nPrecision",
        "gap1124\nSuccess",
        "gap1124\nPrecision",
    ]
    final_delta = [3.758, 7.324, 1.420, 1.818]
    late_delta = [3.127, 6.206, -3.013, -4.424]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    b1 = ax.bar(
        x - width / 2,
        final_delta,
        width,
        label="Epoch60 final",
        color="#2F6B94",
        edgecolor="#173B57",
    )
    b2 = ax.bar(
        x + width / 2,
        late_delta,
        width,
        label="Epoch45-60 mean",
        color="#D28A2E",
        edgecolor="#8D5B17",
    )
    ax.axhline(0, color="#25313A", linewidth=1.0)
    label_bars(ax, b1, offset=0.16, fontsize=8.5)
    label_bars(ax, b2, offset=0.16, fontsize=8.5)
    ax.set_title("Scratch true-time 相对 shuffled-time 的差值", loc="left", fontweight="bold")
    ax.set_ylabel("true - shuffled（百分点）")
    ax.set_xticks(x, labels)
    ax.set_ylim(-6, 9)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(
        0.125,
        0.01,
        "分训含辅助监督冲突；final 与 late window 的分歧不能解释为纯时间因果效应。",
        fontsize=8.5,
        color="#5D6A76",
    )
    return save_figure(fig, CHART_DIR / "scratch_true_shuffled.png")


def chart_current_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, size=10):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color="#1D2730",
            wrap=True,
        )
        return patch

    def arrow(x1, y1, x2, y2, color="#5D6A76", style="-|>"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=style,
                mutation_scale=13,
                linewidth=1.3,
                color=color,
            )
        )

    box(0.2, 3.8, 1.8, 0.8, "历史点云 + 当前点云", "#EEF2F4", "#93A1AD")
    box(0.2, 1.2, 1.8, 0.8, "历史框 + real delta_t", "#EEF2F4", "#93A1AD")
    box(2.5, 3.7, 2.0, 1.0, "SeqTrack3D\norder-time backbone", "#EAF2F7", "#2F6B94")
    box(2.5, 1.1, 2.0, 1.0, "DynamicsEncoder\nvelocity -> d_dyn", "#F7F0DF", "#C79A3B")
    box(5.1, 3.7, 1.8, 1.0, "Observation head\nproposal d_obs", "#EAF2F7", "#2F6B94")
    box(5.1, 1.1, 1.8, 1.0, "Zero-init\nphysical-time adapter", "#F7F0DF", "#C79A3B")
    box(
        7.5,
        2.35,
        2.25,
        1.0,
        "Bounded innovation\nclip(d_dyn-stopgrad(d_obs), R(dt))",
        "#FBEDE6",
        "#D46A3A",
        size=9.2,
    )
    box(10.1, 2.35, 0.9, 1.0, "Box\ndecoder", "#EAF2F7", "#2F6B94")
    box(7.5, 0.35, 3.5, 0.75, "same-checkpoint true / fixed / shuffled controls", "#EEF2F4", "#93A1AD", size=9)

    arrow(2.0, 4.2, 2.5, 4.2)
    arrow(2.0, 1.6, 2.5, 1.6)
    arrow(4.5, 4.2, 5.1, 4.2)
    arrow(4.5, 1.6, 5.1, 1.6)
    arrow(6.0, 3.7, 7.5, 2.9)
    arrow(6.9, 1.6, 7.5, 2.7)
    arrow(6.0, 3.7, 6.0, 2.15, color="#C79A3B")
    arrow(6.0, 2.15, 7.5, 2.65, color="#C79A3B")
    arrow(9.75, 2.85, 10.1, 2.85)
    arrow(9.2, 1.1, 9.2, 2.35, color="#93A1AD", style="->")
    ax.text(0.2, 5.08, "当前 M1/M2 双时钟与有界 proposal correction", fontsize=15, fontweight="bold", color="#173B57")
    ax.text(
        0.2,
        4.82,
        "order clock 保持稳定主干语义；physical clock 只提供增量信息，并接受负对照。",
        fontsize=9.5,
        color="#5D6A76",
    )
    return save_figure(fig, CHART_DIR / "current_architecture.png")


def generate_charts() -> dict[str, Path]:
    configure_matplotlib()
    return {
        "seed_stability": chart_seed_stability(),
        "crop_reachability": chart_crop_reachability(),
        "m0_oracle": chart_m0_oracle(),
        "formal_controls": chart_formal_controls(),
        "scratch_time": chart_scratch_time(),
        "architecture": chart_current_architecture(),
    }


class ReportDoc(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bookmark_id = 0

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        level = None
        if flowable.style.name == "H1":
            level = 0
        elif flowable.style.name == "H2":
            level = 1
        if level is None:
            return
        self._bookmark_id += 1
        key = f"section_{self._bookmark_id}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(
            flowable.getPlainText(), key, level=level, closed=False
        )


def styles():
    base = getSampleStyleSheet()
    result = {}
    result["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        parent=base["Title"],
        fontName="Deng-Bold",
        fontSize=30,
        leading=39,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=7 * mm,
    )
    result["CoverSub"] = ParagraphStyle(
        "CoverSub",
        fontName="Deng",
        fontSize=14,
        leading=22,
        textColor=colors.HexColor("#D7E4EC"),
        spaceAfter=5 * mm,
    )
    result["CoverMeta"] = ParagraphStyle(
        "CoverMeta",
        fontName="Deng",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#D7E4EC"),
    )
    result["H1"] = ParagraphStyle(
        "H1",
        fontName="Deng-Bold",
        fontSize=20,
        leading=28,
        textColor=BLUE,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
        keepWithNext=True,
    )
    result["H2"] = ParagraphStyle(
        "H2",
        fontName="Deng-Bold",
        fontSize=14.5,
        leading=21,
        textColor=BLUE_2,
        spaceBefore=4 * mm,
        spaceAfter=2.5 * mm,
        keepWithNext=True,
    )
    result["H3"] = ParagraphStyle(
        "H3",
        fontName="Deng-Bold",
        fontSize=11.5,
        leading=17,
        textColor=INK,
        spaceBefore=3 * mm,
        spaceAfter=1.5 * mm,
        keepWithNext=True,
    )
    result["Body"] = ParagraphStyle(
        "Body",
        fontName="Deng",
        fontSize=9.6,
        leading=15,
        textColor=INK,
        spaceAfter=2.2 * mm,
        alignment=TA_LEFT,
    )
    result["BodySmall"] = ParagraphStyle(
        "BodySmall",
        fontName="Deng",
        fontSize=8.3,
        leading=12.5,
        textColor=INK,
        spaceAfter=1.6 * mm,
    )
    result["Caption"] = ParagraphStyle(
        "Caption",
        fontName="Deng",
        fontSize=7.7,
        leading=11,
        textColor=SLATE,
        alignment=TA_LEFT,
        spaceBefore=1.2 * mm,
        spaceAfter=3 * mm,
    )
    result["Quote"] = ParagraphStyle(
        "Quote",
        fontName="Deng-Bold",
        fontSize=12,
        leading=19,
        textColor=BLUE,
        leftIndent=4 * mm,
        rightIndent=4 * mm,
        spaceAfter=3 * mm,
    )
    result["Code"] = ParagraphStyle(
        "Code",
        fontName="Courier",
        fontSize=8.1,
        leading=12,
        textColor=INK,
        leftIndent=3 * mm,
        rightIndent=3 * mm,
    )
    result["TableHead"] = ParagraphStyle(
        "TableHead",
        fontName="Deng-Bold",
        fontSize=8.1,
        leading=11,
        textColor=WHITE,
        alignment=TA_CENTER,
    )
    result["TableCell"] = ParagraphStyle(
        "TableCell",
        fontName="Deng",
        fontSize=7.8,
        leading=11.5,
        textColor=INK,
    )
    result["TableCellCenter"] = ParagraphStyle(
        "TableCellCenter",
        fontName="Deng",
        fontSize=7.8,
        leading=11.5,
        textColor=INK,
        alignment=TA_CENTER,
    )
    result["StageLabel"] = ParagraphStyle(
        "StageLabel",
        fontName="Deng-Bold",
        fontSize=8.2,
        leading=12,
        textColor=BLUE,
    )
    result["StageText"] = ParagraphStyle(
        "StageText",
        fontName="Deng",
        fontSize=8.2,
        leading=12.5,
        textColor=INK,
    )
    return result


def cover_page(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#214E6D"))
    canvas.circle(width - 35 * mm, height - 25 * mm, 46 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#2B5D7A"))
    canvas.circle(width - 14 * mm, height - 5 * mm, 30 * mm, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(20 * mm, 43 * mm, 42 * mm, 2.5 * mm, fill=1, stroke=0)
    canvas.restoreState()


def later_page(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7DEE3"))
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
    canvas.setFont("Deng", 7.6)
    canvas.setFillColor(SLATE)
    canvas.drawString(18 * mm, height - 11.5 * mm, "CT-SeqTrack 阶段性进展与成果汇报")
    canvas.drawRightString(
        width - 18 * mm, height - 11.5 * mm, f"截至 {REPORT_DATE}"
    )
    canvas.setStrokeColor(colors.HexColor("#D7DEE3"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("Deng", 7.6)
    canvas.drawString(18 * mm, 9.5 * mm, "技术阶段汇报 | evidence-led")
    canvas.drawRightString(width - 18 * mm, 9.5 * mm, f"{doc.page}")
    canvas.restoreState()


def p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], sty, bullet_color=BLUE_2):
    return ListFlowable(
        [
            ListItem(
                Paragraph(item, sty),
                leftIndent=4 * mm,
                bulletColor=bullet_color,
            )
            for item in items
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=5 * mm,
        bulletFontName="Deng",
        bulletFontSize=7,
        spaceBefore=1 * mm,
        spaceAfter=2.5 * mm,
    )


def callout(text: str, sty, color=BLUE_2, background=BLUE_LIGHT):
    content = Table(
        [[Paragraph(text, sty)]],
        colWidths=[165 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.6, color),
                ("LINEBEFORE", (0, 0), (0, -1), 4, color),
                ("LEFTPADDING", (0, 0), (-1, -1), 7 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ]
        ),
    )
    return content


def stage_card(why: str, built: str, result: str, change: str, sty):
    data = [
        [
            Paragraph("为什么做", sty["StageLabel"]),
            Paragraph(why, sty["StageText"]),
        ],
        [
            Paragraph("构建了什么", sty["StageLabel"]),
            Paragraph(built, sty["StageText"]),
        ],
        [
            Paragraph("实验结论", sty["StageLabel"]),
            Paragraph(result, sty["StageText"]),
        ],
        [
            Paragraph("随后改变", sty["StageLabel"]),
            Paragraph(change, sty["StageText"]),
        ],
    ]
    table = Table(data, colWidths=[27 * mm, 138 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), BLUE_LIGHT),
                ("BACKGROUND", (1, 0), (1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5DC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DEE3")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3 * mm),
            ]
        )
    )
    return table


def data_table(headers, rows, widths, sty, movement_cols=None, small=False):
    movement_cols = movement_cols or []
    header_row = [Paragraph(str(h), sty["TableHead"]) for h in headers]
    table_rows = [header_row]
    cell_style = sty["TableCell"] if not small else sty["BodySmall"]
    for row in rows:
        table_rows.append(
            [
                Paragraph(str(value), cell_style)
                if not isinstance(value, Paragraph)
                else value
                for value in row
            ]
        )
    table = LongTable(table_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4DCE2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.3 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
    ]
    for col in movement_cols:
        for row_index, row in enumerate(rows, start=1):
            raw = str(row[col]).strip()
            if raw.startswith("+"):
                commands.append(
                    ("TEXTCOLOR", (col, row_index), (col, row_index), BLUE_2)
                )
            elif raw.startswith("-") or "No-Go" in raw or "FAIL" in raw:
                commands.append(
                    ("TEXTCOLOR", (col, row_index), (col, row_index), ORANGE)
                )
    table.setStyle(TableStyle(commands))
    return table


def report_image(path: Path, max_width=165 * mm, max_height=93 * mm):
    if not path.exists():
        raise FileNotFoundError(path)
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def figure_block(path: Path, caption: str, sty, max_height=93 * mm):
    return [
        report_image(path, max_height=max_height),
        Paragraph(caption, sty["Caption"]),
    ]


def section_start(story, title, subtitle, sty, page_break=True):
    if page_break:
        story.append(PageBreak())
    story.append(Paragraph(title, sty["H1"]))
    if subtitle:
        story.append(Paragraph(subtitle, sty["Body"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=GOLD, spaceAfter=4 * mm))


def build_story(chart_paths: dict[str, Path]):
    sty = styles()
    story = []

    # Cover
    story.append(Spacer(1, 46 * mm))
    story.append(Paragraph("CT-SeqTrack", sty["CoverTitle"]))
    story.append(Paragraph("阶段性进展与成果汇报", sty["CoverTitle"]))
    story.append(
        Paragraph(
            "从固定帧序列学习到 variable-rate 3D SOT："
            "时间建模、失败诊断、负对照与 bounded proposal correction",
            sty["CoverSub"],
        )
    )
    story.append(Spacer(1, 12 * mm))
    cover_status = Table(
        [
            [
                Paragraph("<b>M2 tracking signal</b><br/>POSITIVE", sty["CoverMeta"]),
                Paragraph("<b>Physical-time causal claim</b><br/>NO-GO", sty["CoverMeta"]),
                Paragraph("<b>Method attribution</b><br/>HOLD", sty["CoverMeta"]),
            ]
        ],
        colWidths=[52 * mm, 52 * mm, 52 * mm],
    )
    cover_status.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#285979")),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#7A4937")),
                ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#4A5E6E")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D7E4EC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#D7E4EC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ]
        )
    )
    story.append(cover_status)
    story.append(Spacer(1, 23 * mm))
    story.append(
        Paragraph(
            f"汇报时间：截至 {REPORT_DATE}<br/>"
            "研究基线：SeqTrack3D | 数据：nuScenes-mini Car（当前阶段）<br/>"
            "报告口径：只把有完整实验或明确门禁的内容列为成果；候选方法单独标记。",
            sty["CoverMeta"],
        )
    )
    story.append(PageBreak())

    # Technical summary
    section_start(
        story,
        "技术摘要：项目方向成立，但主张已经收窄",
        "本页先给出当前结论，后续按时间顺序解释每一次构建、实验与路线调整。",
        sty,
        page_break=False,
    )
    story.append(
        callout(
            "<b>当前最准确的一句话：</b>CT-SeqTrack 已证明 M2 训练得到的 tracker "
            "在 standard 与 gap1124 相对 matched A1 有稳定正信号；但同 checkpoint 的 "
            "true/fixed/shuffled 表明正确 physical time 没有形成因果优势，"
            "因此当前成果应表述为 variable-rate 协议、失败诊断与待归因的通用 proposal correction。",
            sty["Body"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    summary_rows = [
        ["工程地基", "P0-P5、M1/M2 E0-E6 已完成", "时间字段、双视图、dynamics、Gate、shared-SE(2)、strict fallback 与 provenance 已形成完整链路"],
        ["最强正结果", "standard +4.133/+9.445；gap1124 +2.279/+4.143", "M2 true 相对 matched A1，tracklet bootstrap 的 Success/Precision 95% CI 均为正"],
        ["最关键负结果", "true 不优于 fixed/shuffled", "模型会读取时间并改变预测，但正确 alignment 不是当前涨分来源"],
        ["当前论文价值", "协议/诊断最稳，方法归因待补", "bounded innovation 有潜力；timestamp-conditioned M3/M4 不应直接晋级"],
    ]
    story.append(
        data_table(
            ["维度", "当前状态", "解释"],
            summary_rows,
            [31 * mm, 53 * mm, 81 * mm],
            sty,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "<b>汇报时应把“涨分”和“时间因果”分开。</b>"
            "前者已经出现，后者已经在冻结负对照中失败。这个区分不是削弱成果，"
            "而是说明项目已从单纯追分进入可审计的机制研究阶段。[S17][S18]",
            sty["Body"],
        )
    )

    story.append(Paragraph("阅读路径", sty["H2"]))
    story.append(
        bullets(
            [
                "<b>第一部分：</b>为什么从 SeqTrack3D 出发，以及最初创新点是什么。",
                "<b>第二部分：</b>从 P0-P5 到 TWC/Gate/HTV 的逐步构建和逐轮否证。",
                "<b>第三部分：</b>为什么转向 crop/trajectory diagnosis，并形成 M0-M2。",
                "<b>第四部分：</b>当前涨分、尚未闭环的问题与下一阶段优先级。",
            ],
            sty["Body"],
        )
    )

    # Definitions
    section_start(
        story,
        "范围与指标：所有数字必须先明确比较口径",
        "当前主要证据来自 nuScenes-mini Car；Success 与 Precision 均使用 one-pass tracking 评测。",
        sty,
    )
    definition_rows = [
        ["A1-order", "SeqTrack3D 主干保留固定 order-time；关闭 dynamics/TWC/Gate", "稳定观测基线"],
        ["A2-order-dyn", "order-time 主干 + 真实 delta_t 的 feature-concat DynamicsEncoder", "旧时间接入方法，已 No-Go"],
        ["M2", "zero-init adapter + shared-SE(2) canonical supervision + bounded proposal innovation", "当前待归因方法"],
        ["standard", "约 0.5 s 固定 keyframe cadence", "正常条件 guardrail，delta_t 可辨识性低"],
        ["gap1124", "同一 tracklet 内按 1-1-2-4 gap 模式重采样", "强不规则 cadence"],
        ["random20", "随机丢帧约 20%，保留固定 manifest", "温和不规则 cadence"],
        ["true/fixed/shuffled", "同 endpoint、几何输入和 checkpoint，仅替换 effective time", "物理时间因果负对照"],
    ]
    story.append(
        data_table(
            ["术语", "定义", "实验角色"],
            definition_rows,
            [28 * mm, 88 * mm, 49 * mm],
            sty,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "<b>解释原则：</b>standard 上涨分只能说明 tracker 更强；"
            "只有同 checkpoint 下 true 同时优于 fixed 与 shuffled，"
            "才允许把涨分归因于正确物理时间。",
            sty["Body"],
            color=GOLD,
            background=GOLD_LIGHT,
        )
    )
    story.append(Paragraph("比较层级", sty["H2"]))
    story.append(
        bullets(
            [
                "<b>描述性比较：</b>不同历史 run 的最终分数，只能用于方向参考。",
                "<b>matched tracking 比较：</b>同 endpoint 的 M2 与 A1，可说明总体 tracking signal。",
                "<b>同 checkpoint 时间干预：</b>同一模型 true/fixed/shuffled，可回答 physical-time causal claim。",
                "<b>离线 oracle：</b>只说明存在机制空间，不能直接等价为在线 Success/Precision。",
            ],
            sty["Body"],
        )
    )

    # Timeline overview
    section_start(
        story,
        "时间线总览：创新点不是一次确定，而是被实验逐步筛选",
        "下面的时间线也是汇报的主逻辑。每一次 No-Go 都对应一次更严格的问题重定义。",
        sty,
    )
    timeline_rows = [
        ["2026-05-26", "P0-P2", "打通真实 timestamp 与 TimeEncoding", "工程 PASS"],
        ["2026-05-26 至 05-27", "P3-P5", "Dynamics、TWC、Observability Gate", "完成实现，等待消融"],
        ["2026-06", "A1/A2 系列", "发现 raw real-time 主干崩坏，恢复 order-time", "创新从“替换时间”改为“双时钟”"],
        ["2026-07-08 至 07-16", "稳定性/HTV/TWC 修复", "seed 风险、强 cadence 退化、TWC 坐标污染", "停止直接堆结构"],
        ["2026-07-17 至 07-20", "P0-A/B/C", "crop、递归历史、reliability、时间负对照", "Gate/raw-CV/A2 true-dt No-Go"],
        ["2026-07-21", "TWC A/B/C + M0", "TWC 因果拆分、proposal oracle、candidate 伪动力学", "解锁 shared-SE(2) + bounded innovation"],
        ["2026-07-22 至 07-23", "M1/M2 E0-E6 + R1/R2/R3", "完成 formal 工程门禁与三组训练", "standard tracking signal positive"],
        ["2026-07-24", "正式 controls + scratch", "standard/gap1124 八组和 true/shuffled 分训", "physical-time No-Go；attribution Hold"],
        ["2026-07-24", "M3/M4 工程稿", "非对称 path distillation 与 fixed filter/tube", "仅原型，未形成实验贡献"],
    ]
    story.append(
        data_table(
            ["时间", "阶段", "核心工作", "阶段结论"],
            timeline_rows,
            [28 * mm, 31 * mm, 69 * mm, 37 * mm],
            sty,
        )
    )

    # Stage 1
    section_start(
        story,
        "阶段 1（2026-05-26）：先把真实时间变成一等输入",
        "起点不是立即设计复杂模型，而是先证明训练与测试链路真的携带 physical timestamp。",
        sty,
    )
    story.append(
        stage_card(
            "SeqTrack3D 用固定帧序列和伪时间表达历史，无法区分正常 2Hz、跳帧和长 gap。",
            "训练/测试统一输出 timestamps、delta_t、delta_T、current_timestamp、current_delta_t；"
            "point time 与 box-corner time 共用 scalar-preserving TimeEncoding，支持 raw/MLP/Fourier。",
            "真实 batch 字段、CPU forward、GPU loss 与 2-step train smoke 均通过；P0-P2 工程闭环。",
            "具备了研究 variable-rate 3D SOT 的最低基础，但此时还没有任何涨分结论。",
            sty,
        )
    )
    story.append(Paragraph("最初的创新设想", sty["H2"]))
    story.append(
        bullets(
            [
                "<b>连续时间状态转移：</b>由 delta_t 解释历史框差分和当前 query gap。",
                "<b>时间重采样一致性：</b>不同历史采样路径到同一 endpoint 应给出一致预测。",
                "<b>可观测性融合：</b>观测可靠时信点云，稀疏/遮挡时信 dynamics prior。",
                "<b>连续时间评测：</b>除标准主表外，加入 gap、drop、sparse 和 re-appearance。",
            ],
            sty["Body"],
        )
    )
    story.append(
        callout(
            "<b>这一阶段的成果是“输入契约创新”，不是性能创新。</b>"
            "后续所有模块都必须先证明读取的是真实、可复现、训练测试一致的时间字段。[S1][S2][S3]",
            sty["Body"],
        )
    )

    # Stage 2
    section_start(
        story,
        "阶段 2（2026-05-26 至 05-27）：构建 Dynamics、TWC 与 Gate",
        "这是最初 CT-SeqTrack full model 的工程实现阶段。",
        sty,
    )
    story.append(
        stage_card(
            "仅有时间字段不够，需要让时间参与运动估计、跨采样一致性和观测/先验融合。",
            "P3 DynamicsEncoder 计算 velocity/angular velocity 并预测 displacement；"
            "P4 构造 A/B 历史视图与 TWC loss；P5 用点数、前景概率、历史有效率和 gap 计算融合权重。",
            "三个模块的 forward/loss/训练步均通过，且默认关闭保证 baseline 兼容。",
            "进入正式消融，但创新点仍是并列堆叠，尚未形成单一主轴。",
            sty,
        )
    )
    story.append(Paragraph("P3 的第一版逻辑", sty["H2"]))
    story.append(
        callout(
            "v_i = (c_i - c_(i-1)) / delta_t_i<br/>"
            "omega_i = wrap(theta_i - theta_(i-1)) / delta_t_i<br/>"
            "d_dyn = velocity_pred * current_delta_t",
            sty["Code"],
            color=GOLD,
            background=GOLD_LIGHT,
        )
    )
    story.append(
        Paragraph(
            "此时 Dynamics 以 feature concat 接入 coarse motion head；TWC 使用对称双路监督；"
            "Gate 在 observation feature 与 z_dyn 之间做 softmax 融合。"
            "工程上完整，但三者都存在潜在的训练分布与语义混杂，后续实验逐一暴露出来。[S3][S4]",
            sty["Body"],
        )
    )

    # Stage 3
    section_start(
        story,
        "阶段 3（2026-06）：直接替换主干时间语义失败",
        "第一次关键转向：问题不在编码器形式，而在 SeqTrack3D 主干依赖原有 order-time 语义。",
        sty,
    )
    story.append(
        stage_card(
            "需要确认早期 P5 full 崩坏是 Gate、Dynamics 还是 real-time token 导致。",
            "拆出 A1-raw、A1-pseudo、A1-MLP、A1-Fourier、scaled real-time，并逐项对比。",
            "baseline 约 50.99/59.96；A1-raw 约 28.3/27.4；pseudo 恢复到 48.3/52.3；"
            "MLP/Fourier 仍明显退化。",
            "停止把真实秒数直接替换主干 token；主干恢复 order-time，physical time 只进入增量分支。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>这张图回答“换一种时间编码能不能救回来”。</b>"
            "结果显示 pseudo/order 语义比 raw real-time 更稳定，而 MLP/Fourier 并未解决退化，"
            "因此创新点从“更强时间编码”改成“保留顺序时钟 + 独立物理时钟”。",
            sty["Body"],
        )
    )
    early_chart = ROOT / "compare_results" / "figures" / "bar_charts" / "a1_time_encoding_best_final_summary.png"
    story.extend(
        figure_block(
            early_chart,
            "图 1  A1 时间编码消融。final/best 都显示 real-time 主干接入不稳定。来源：[S5]。",
            sty,
            max_height=82 * mm,
        )
    )
    story.append(
        callout(
            "<b>创新点第一次修改：</b>不再宣称“把 SeqTrack3D 改成真实时间 token”就是连续时间建模；"
            "改为 dual-clock 思路：order clock 保护主干，physical clock 只提供受控增量。",
            sty["Body"],
            color=GOLD,
            background=GOLD_LIGHT,
        )
    )

    # Stage 4
    section_start(
        story,
        "阶段 4（2026-06 至 07-08）：A2 出现涨分，但稳定性不足",
        "恢复 A1-order 后，A2-order-dyn 在 seed42 的 Precision 上出现第一批清楚正信号。",
        sty,
    )
    story.append(
        stage_card(
            "既然主干 real-time 有害，需要检查真实 delta_t 是否能作为独立 motion prior 发挥作用。",
            "构建 A1-order、A2-order-dyn，并补 cand1、displacement supervision、seed43/44 和 conf-res 复测。",
            "A1-order 51.23/57.86；A2 seed42 50.96/63.31，但 seed43 为 23.64/23.77，"
            "seed44 仅 46.90/52.62。",
            "不能把 seed42 的 Precision 增益写成稳定结论；研究重点转向协议依赖、candidate 噪声和机制诊断。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>跨 seed 图说明“出现过涨分”不等于“方法已成立”。</b>"
            "seed42 的 Precision 比 A1 高，但 seed43 发生塌陷，seed44 也没有回到 seed42 水平。",
            sty["Body"],
        )
    )
    story.extend(
        figure_block(
            chart_paths["seed_stability"],
            "图 2  A2-order-dyn 的 seed42/43/44 最终指标。来源：[S6]。",
            sty,
            max_height=79 * mm,
        )
    )
    story.append(Paragraph("cand1 与 displacement 的作用", sty["H2"]))
    story.append(
        bullets(
            [
                "<b>cand1：</b>60 epoch 下明显退化，但 optimizer steps 约为 cand4 的 1/4，不能据此删除 nonzero candidates。",
                "<b>displacement supervision：</b>与 A2 基本持平，Precision 小幅上升；说明不伤主线，但不是主要解释。",
                "<b>conf-res Gate：</b>旧 best 很高，best-e14 复测却只有 28.06/37.70，不能作为确认收益。",
            ],
            sty["Body"],
        )
    )

    # Stage 5 TWC
    section_start(
        story,
        "阶段 5（2026-05-27 至 07-21）：TWC 从核心创新降为机制诊断",
        "TWC 的研究过程包含一次重要工程纠错和一次同提交因果拆分。",
        sty,
    )
    story.append(
        stage_card(
            "希望同一 endpoint 的不同历史采样路径保持一致，从而提升 path robustness。",
            "构造 A:[1,2,3] 与 B:[1,3,5] 双视图；修复 shared candidate offset、coordinate anchor、"
            "point-sampling seed 和 current XYZ 检查；最终做 A/B/C 同提交实验。",
            "旧 TWC 存在 nonzero candidate 坐标污染，早期归因撤回。修复后 B-A=-15.30/-24.18，"
            "C-B=+8.31/+11.74，但 C-A=-7.00/-12.44。",
            "对称 TWC 不能作为主方法；保留“部分修复 paired-view 退化”的机制事实，并提出非对称 EMA teacher 方案。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>实验必须拆成三个臂：</b>A 是 single-view，B 是 paired views 但 TWC 权重为 0，"
            "C 是 paired views + corrected TWC。只有 C-B 才是 consistency 的净效应；"
            "C-A 才是部署价值。",
            sty["Body"],
        )
    )
    twc_chart = ROOT / "compare_results" / "figures" / "delta_charts" / "twc_abc_seed42_effect_deltas.png"
    story.extend(
        figure_block(
            twc_chart,
            "图 3  TWC A/B/C 效应分解。TWC 净效应为正，但不足以追回 paired-view 主任务损失。来源：[S13]。",
            sty,
            max_height=84 * mm,
        )
    )
    story.append(
        callout(
            "<b>创新点第二次修改：</b>放弃对称 0.5*(L_A+L_B)+L_twc，"
            "候选 M3 改为 canonical EMA teacher -> irregular student，第一轮 beta=0，"
            "避免困难 B 路的真值监督破坏 A 分布。",
            sty["Body"],
            color=GOLD,
            background=GOLD_LIGHT,
        )
    )

    # Stage 6 Gate
    section_start(
        story,
        "阶段 6（2026-05-27 至 07-20）：Observability Gate 未通过独立验证",
        "Gate 的想法合理，但开发集正信号没有转化为可部署的独立验证结果。",
        sty,
    )
    story.append(
        stage_card(
            "稀疏或遮挡时当前点云不可靠，希望模型自动提高 dynamics 权重。",
            "P5 Gate、gate-safe、conf-res；随后将问题前移为 pre-crop risk prediction，"
            "构建 P0-B3 passive reliability 与 P0-B4 frozen observation_v1。",
            "P0-B3 的 all-13 AUROC 为 0.857/0.787/0.785，但主要信号来自 previous observation；"
            "P0-B4 gap/burst AUROC 仅 0.680/0.712，recall 0.568/0.609，低于 0.75/0.70 门槛。",
            "hand-crafted Gate、reliability-controlled anchor、active dual-anchor 和 selector 全部停止。",
            sty,
        )
    )
    reliability_rows = [
        ["P0-B3 standard", "0.857 / 0.742", "开发集全特征 trigger", "通过原门槛，但不能外推"],
        ["P0-B3 gap1124", "0.787 / 0.660", "强协议", "raw delta_t 导致失准"],
        ["P0-B3 burst-drop", "0.785 / 0.671", "强协议", "上一观测质量更有预测力"],
        ["P0-B4 gap1124", "0.680 / 0.282", "独立 mini_val", "No-Go"],
        ["P0-B4 burst-drop", "0.712 / 0.328", "独立 mini_val", "No-Go"],
    ]
    story.append(
        data_table(
            ["实验", "AUROC / AUPRC", "角色", "结论"],
            reliability_rows,
            [36 * mm, 34 * mm, 45 * mm, 50 * mm],
            sty,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "<b>这条失败链的意义：</b>项目没有在 mini_val 上继续重调 feature/threshold，"
            "而是保留预注册 No-Go。它证明了“能预测风险”与“能形成互补 proposal”是两件事。[S10][S11]",
            sty["Body"],
            color=ORANGE,
            background=ORANGE_LIGHT,
        )
    )

    # Stage 7 HTV
    section_start(
        story,
        "阶段 7（2026-07-08 至 07-16）：建立 variable-rate/HTV 协议",
        "当 fixed-step 主表无法解释时间机制时，研究主战场转向同一 tracklet 内的不规则 cadence。",
        sty,
    )
    story.append(
        stage_card(
            "standard delta_t 接近常数，模型可以把时间函数吸收到普通权重中，难以证明 physical time。",
            "构建 gap1124、burst-drop、random20，冻结 cadence manifest、endpoint selection 与训练步。",
            "旧 A2 相对 A1：gap1124 -4.01/-9.55，burst-drop -7.45/-14.40，"
            "random20 +9.09/+14.23。",
            "“越高时变越有效”的假设不成立；需要区分 cadence、crop 可达性、历史可靠性与模型接入方式。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>HTV 图揭示明显 protocol dependence。</b>"
            "温和 random20 有正信号，强 gap/burst 反而退化，说明旧 feature-concat dynamics "
            "不是对不规则采样的通用解法。",
            sty["Body"],
        )
    )
    htv_chart = ROOT / "compare_results" / "figures" / "delta_charts" / "htv_6runs_a2_minus_a1_deltas.png"
    story.extend(
        figure_block(
            htv_chart,
            "图 4  旧 A2-order-dyn 在三种 HTV 协议下相对 A1 的差值。来源：[S7]。",
            sty,
            max_height=84 * mm,
        )
    )
    story.append(
        callout(
            "<b>创新点第三次修改：</b>HTV 不再作为“首次 skip-frame”贡献；"
            "真正可辨识的边界改为 within-track irregularity、matched endpoint、"
            "one-checkpoint 跨 cadence 与 true/fixed/shuffled 干预。",
            sty["Body"],
            color=GOLD,
            background=GOLD_LIGHT,
        )
    )

    # Stage 8 crop diagnosis
    section_start(
        story,
        "阶段 8（2026-07-17）：发现核心失败常发生在模型 forward 之前",
        "这一步把问题从“末端怎么融合”前移到“目标有没有进入 search crop”。",
        sty,
    )
    story.append(
        stage_card(
            "bounded residual 即使更准确，也无法追回已经离开搜索区域的目标。",
            "P0-A/P0-B 比较 base crop、2x expanded 与 GT-history constant-velocity recenter，"
            "并按 gap、位移和点数分桶。",
            "base recall 为 85.41/76.78/77.72；强协议 2x expanded 仅 89.08/87.65，"
            "GT-history CV 达 98.96/99.05，且点数成本接近 base。",
            "停止只在输出端调 residual；轨迹方法若继续，必须先证明 GT-free predicted-history 的 search support 互补性。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>固定扩大 crop 既昂贵又不足。</b>"
            "强协议下 2x expanded 平均点数约为 base 的 5.2 倍，却仍有约 11%-12% 目标点未召回；"
            "GT-history CV 接近 99%，说明正确移动搜索中心比无脑扩大范围更关键。",
            sty["Body"],
        )
    )
    story.extend(
        figure_block(
            chart_paths["crop_reachability"],
            "图 5  不同 crop 策略的目标点召回率。GT-history CV 是 oracle，不能写成在线收益。来源：[S8]。",
            sty,
            max_height=82 * mm,
        )
    )
    story.append(Paragraph("递归预测历史为什么没有直接成功", sty["H2"]))
    story.append(
        bullets(
            [
                "previous-A1 crop recall：69.69% / 63.73% / 63.24%。",
                "predicted-history CV recall：72.61% / 66.38% / 66.27%，只提高 2.65-3.03 pp。",
                "上一预测误差 &lt;=4 m 时，CV recall 可达约 97%-99%；误差 &gt;4 m 后降到约 1%。",
                "结论：trajectory prior 更适合作为预防性第二支持，而不是从严重漂移状态中单独恢复。",
            ],
            sty["Body"],
        )
    )

    # Stage 9 P0-C
    section_start(
        story,
        "阶段 9（2026-07-20 至 07-21）：用时间负对照否定旧 A2 的 physical-time 主张",
        "模型“读取了时间”与“正确时间有用”必须通过同 checkpoint 干预区分。",
        sty,
    )
    story.append(
        stage_card(
            "A2 在不同协议分别训练评测，无法证明一个模型是否真正利用正确的 delta_t。",
            "P0-C 拆分 train/val/test cadence，构建 stable-token manifest 与 offline shuffled mapping；"
            "同一冻结 A2 checkpoint 评测 true/fixed/shuffled。",
            "true-fixed +0.438/+0.523；true-shuffled -0.123/+0.056；"
            "tracklet bootstrap CI 跨 0，未达到 +0.5/+1.0 门槛。",
            "feature-concat A2 的 true-dt promotion No-Go；协议资产保留，方法解释停止。",
            sty,
        )
    )
    p0c_rows = [
        ["true", "55.225", "66.885", "使用真实 effective time"],
        ["fixed", "54.787", "66.362", "统一为 0.5 s"],
        ["shuffled", "55.348", "66.830", "保持 gap 边际分布，打乱对应关系"],
        ["true-fixed", "+0.438", "+0.523", "未过 promotion margin"],
        ["true-shuffled", "-0.123", "+0.056", "正确 alignment 无稳定优势"],
    ]
    story.append(
        data_table(
            ["P0-C arm", "Success", "Precision", "解释"],
            p0c_rows,
            [34 * mm, 30 * mm, 30 * mm, 71 * mm],
            sty,
            movement_cols=[1, 2],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "<b>研究方法上的进步：</b>从这一步起，任何 explicit-dt 模块都必须接受 "
            "same-checkpoint true/fixed/shuffled。没有这个对照，普通 benchmark 涨分不能被写成时间机制。",
            sty["Body"],
        )
    )

    # Stage 10 M0 pivot
    section_start(
        story,
        "阶段 10（2026-07-21）：M0 oracle 促成 proposal innovation 重构",
        "旧 bounded residual 量级近乎为零，同时存在“两个完整位移相加”的语义歧义。",
        sty,
    )
    story.append(
        stage_card(
            "需要先回答 d_dyn 是否真的包含 d_obs 之外的信息，以及 candidate jitter 是否污染运动标签。",
            "M0-3 在 crop-reachable cohort 导出 d_obs/d_dyn/d_gt 与 oracle blend；"
            "M0-4 审计 candidate0/1/2/3 的伪速度、伪加速度和 matched proposal penalty。",
            "d_obs/d_dyn/oracle mean error 为 1.349/0.309/0.232 m；d_dyn 在 81.31% endpoint 更优。"
            "非零 candidate 伪速度 P50 0.611 m/s、伪加速度 2.128 m/s^2。",
            "冻结 shared world-SE(2) canonical supervision；把完整位移相加改为 bounded proposal innovation。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>M0-3 证明“插值空间存在”。</b>"
            "Dynamics proposal 明显优于 observation proposal，oracle 再进一步降低误差，"
            "因此允许进入一次预注册的 M2 工程；但这是训练 sampler 上的离线 cohort，"
            "不能直接等价为在线 tracking 涨分。",
            sty["Body"],
        )
    )
    story.extend(
        figure_block(
            chart_paths["m0_oracle"],
            "图 6  M0-3 proposal 误差。dynamics-only 与 oracle 均支持互补性。来源：[S14]。",
            sty,
            max_height=75 * mm,
        )
    )
    story.append(
        callout(
            "<b>创新点第四次修改：</b><br/>"
            "旧：d_final = d_obs + alpha*d_dyn<br/>"
            "新：innovation = clip(d_dyn - stopgrad(d_obs), R(delta_t)); "
            "d_final = d_obs + alpha*innovation",
            sty["Body"],
            color=GOLD,
            background=GOLD_LIGHT,
        )
    )

    # Stage 11 M1/M2 architecture
    section_start(
        story,
        "阶段 11（2026-07-21 至 07-22）：形成当前 M1/M2 方法主轴",
        "方法不再由 Dyn/TWC/Gate 三个并列模块组成，而是围绕 dual-clock + bounded correction 收敛。",
        sty,
    )
    story.append(
        stage_card(
            "需要在不破坏 A1 的前提下引入 physical-time 信息，并确保 candidate augmentation 物理一致。",
            "M1：shared world-SE(2)、canonical label、zero-init adapter；"
            "M2：proposal innovation、time-dependent radius、strict fallback、共享 warmup=5。",
            "E0-E5 在 clean commit 9a0b26d 通过；E6 冻结 alpha=0.75、"
            "R(dt)=min(0.5+0.5dt,2.0)、75720 steps、last checkpoint 与 provenance。",
            "具备启动唯一 seed42 formal run 的工程条件；但 engineering PASS 不代表性能或因果 PASS。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>当前结构有两个时钟。</b>"
            "order clock 继续服务 SeqTrack3D 的 point/corner backbone；physical clock 只进入 "
            "DynamicsEncoder、zero-init adapter 与 R(delta_t)。任何一条增量路径关闭时都要求严格回到 A1。",
            sty["Body"],
        )
    )
    story.extend(
        figure_block(
            chart_paths["architecture"],
            "图 7  当前 M1/M2 架构。true/fixed/shuffled 只改变 effective time，不改变 endpoint 或几何输入。来源：[S15][S17]。",
            sty,
            max_height=84 * mm,
        )
    )
    gate_rows = [
        ["E0", "默认回归", "新功能默认关闭，旧 A1 路径不回归", "PASS"],
        ["E1-E2", "几何/标签不变量", "shared-SE(2) 与 canonical dynamics label", "PASS"],
        ["E3", "公式不变量", "zero/invalid/empty/warmup 精确回到 A1", "PASS"],
        ["E4-E5", "数值/可训练性", "三协议 finite、2-step、非零梯度、有界修正", "PASS"],
        ["E6", "可复现性", "唯一配置、步数、checkpoint、manifest、hash", "PASS"],
    ]
    story.append(
        data_table(
            ["Gate", "检查目标", "证据", "状态"],
            gate_rows,
            [22 * mm, 36 * mm, 82 * mm, 25 * mm],
            sty,
        )
    )

    # Stage 12 R1/R2/R3
    section_start(
        story,
        "阶段 12（2026-07-23）：M2 在 standard 上出现最强涨分",
        "三组训练完整结束，但归因缺口同时暴露出来。",
        sty,
    )
    story.append(
        stage_card(
            "需要确认 full M2 是否能从 A1 初始化或 scratch 训练，并与 shared-SE(2) W0 比较。",
            "R1=A1-init M2；R2=scratch M2；R3=scratch shared-SE(2) W0；"
            "三组同 commit、seed42、60 epoch、75720 steps、last checkpoint。",
            "A1 51.23/57.86；R1 55.30/67.18；R2 53.32/62.50；R3 29.00/28.02。"
            "R1/R2 的 late mean 也保持正向。",
            "记录 M2 STANDARD SIGNAL POSITIVE，但 R1 混入额外 60 epoch，R3 又不是有效历史 A1 代理，归因仍 Hold。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>R1 是当前最高分，R2 说明 M2 可从 scratch 工作。</b>"
            "但 R3 严重塌陷说明 shared-SE(2) 与原模型路径存在强交互，"
            "不能把 R2-R3 的 24/34 点全部写成 M2 净收益。",
            sty["Body"],
        )
    )
    story.append(
        callout(
            "<b>此时最重要的两个缺失对照：</b>"
            "A1-init W0 continuation（排除额外训练）和 current-code legacy-candidate W0（解释 R3 塌陷）。",
            sty["Body"],
            color=ORANGE,
            background=ORANGE_LIGHT,
        )
    )

    # Stage 13 formal controls
    section_start(
        story,
        "阶段 13（2026-07-24）：正式 controls 确认涨分，也否定时间因果",
        "这是目前最重要、证据等级最高的一组结果。",
        sty,
    )
    story.append(
        stage_card(
            "standard 涨分仍不能回答正确 physical time 是否造成收益。",
            "固定 R1 epoch60 checkpoint，在 standard/gap1124 各跑 true/fixed/shuffled，"
            "同时导出 matched A1；验证 89/89 artifact hash、endpoint identity 与原始 CSV 指标。",
            "M2-A1：standard +4.133/+9.445，gap1124 +2.279/+4.143，tracklet CI 均为正。"
            "但所有 true-fixed/shuffled 差值接近 0 或为负，CI 均跨 0。",
            "状态更新为 tracking signal positive / physical-time causal claim No-Go / method attribution Hold。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>这张图必须作为汇报的核心证据。</b>"
            "左侧两个 M2-A1 柱说明 tracker 确实变强；后四组时间干预接近零，"
            "说明当前正信号不能解释为正确物理秒数带来的收益。",
            sty["Body"],
        )
    )
    story.extend(
        figure_block(
            chart_paths["formal_controls"],
            "图 8  R1 matched tracking 增益与 same-checkpoint 时间负对照。来源：[S17]。",
            sty,
            max_height=86 * mm,
        )
    )
    formal_rows = [
        ["standard M2-A1", "+4.133", "+9.445", "[1.920,5.454] / [4.486,10.634]", "PASS"],
        ["gap1124 M2-A1", "+2.279", "+4.143", "[0.687,3.940] / [1.452,6.022]", "PASS"],
        ["standard true-fixed", "+0.031", "-0.010", "均跨 0", "FAIL causal gate"],
        ["standard true-shuffled", "+0.068", "+0.085", "均跨 0", "FAIL causal gate"],
        ["gap1124 true-fixed", "-0.127", "+0.014", "均跨 0", "FAIL causal gate"],
        ["gap1124 true-shuffled", "-0.318", "-0.209", "均跨 0", "FAIL causal gate"],
    ]
    story.append(
        data_table(
            ["comparison", "dSuccess", "dPrecision", "tracklet 95% CI", "判定"],
            formal_rows,
            [38 * mm, 22 * mm, 24 * mm, 45 * mm, 36 * mm],
            sty,
            movement_cols=[1, 2, 4],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        bullets(
            [
                "时间路径并未失活：gap1124 约 90.8% 非初始 endpoint 应用了 innovation。",
                "true 将平均 innovation radius 从 fixed 的 0.750 m 增至 0.962 m。",
                "true/fixed/shuffled 会改变递归预测，但正确 alignment 没有更高 Accuracy。",
                "M2-A1 在四个 real-dt 桶都为正；true-shuffled 的 Success 在四桶都为负。",
            ],
            sty["Body"],
        )
    )

    # Stage 14 latest scratch
    section_start(
        story,
        "阶段 14（2026-07-24 最新）：scratch 分训出现部分时间信号，但不能复活主张",
        "这是根文档之后新增的最新补充实验，必须和 formal same-checkpoint 结果一起解释。",
        sty,
        page_break=False,
    )
    story.append(
        stage_card(
            "希望检查 true-time 与 shuffled-time 从 scratch 学习时是否形成稳定差异，以及强 HTV 是否放大这种差异。",
            "random20/gap1124 各训练 true 与 shuffled 两组，四组均 clean 473738f、seed42、60 epoch。",
            "random20 final +3.758/+7.324，late +3.127/+6.206；"
            "gap1124 final +1.420/+1.818，但 late -3.013/-4.424，曲线多数点由 shuffled 占优。",
            "只记录 random20 partial learnability signal；HTV amplification 不支持；physical-time No-Go 不变。",
            sty,
        )
    )
    story.append(
        Paragraph(
            "<b>final 与 late window 的方向分裂是关键。</b>"
            "random20 的 true 在 12 个验证点中赢 10/12、11/12；gap1124 只赢 3/12、2/12。"
            "更强的时间置乱没有带来更稳定 true 优势。",
            sty["Body"],
        )
    )
    story.extend(
        figure_block(
            chart_paths["scratch_time"],
            "图 9  Scratch true-time 相对 shuffled-time 的 final 与 late delta。来源：[S18]。",
            sty,
            max_height=80 * mm,
        )
    )
    story.append(
        callout(
            "<b>重要混杂：</b>当前 displacement_pred = velocity_pred * delta_t_effective，"
            "但 velocity/displacement label 仍由真实时间定义。shuffled 分训无法同时满足两项辅助监督，"
            "因此它是 learnability stress test，不是纯时间因果实验。",
            sty["Body"],
            color=ORANGE,
            background=ORANGE_LIGHT,
        )
    )

    # Innovation evolution
    section_start(
        story,
        "创新点演化：从四个并列模块收敛为一个可审计主轴",
        "创新点的修改不是换题，而是通过负对照不断去掉无法被证据支持的部分。",
        sty,
    )
    innovation_rows = [
        ["最初", "真实时间编码替换主干 token", "A1-raw/MLP/Fourier 退化", "停止替换；保留 order clock"],
        ["最初", "feature-concat DynamicsEncoder", "seed/protocol 不稳定；P0-C No-Go", "仅保留为 proposal source"],
        ["最初", "对称 TWC", "C-B 正，但 C-A 明显负", "改为 time-agnostic 非对称 path distillation 候选"],
        ["最初", "Observability Gate", "P0-B4 独立验证 No-Go", "不复活 hand-crafted Gate"],
        ["中期", "末端 bounded residual", "默认 applied 约 1e-7 m；且重复完整位移", "改为 d_dyn-d_obs innovation"],
        ["当前", "shared-SE(2)+dual clock+bounded innovation", "tracking 正信号；physical-time No-Go", "转入 adapter/innovation/continuation 归因"],
        ["候选", "M3 endpoint path distillation", "代码原型，无性能结果", "仅 time-agnostic 重新预注册后可测"],
        ["候选", "M4 fixed filter + trajectory tube", "代码原型，无 tube oracle/在线结果", "必须先过 predicted-history tube oracle"],
    ]
    story.append(
        data_table(
            ["阶段", "创新设想", "证据", "当前处理"],
            innovation_rows,
            [18 * mm, 43 * mm, 50 * mm, 54 * mm],
            sty,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "<b>当前最可防御的创新组合：</b>"
            "matched within-track variable-rate protocol + same-checkpoint time interventions + "
            "crop/recursive failure diagnosis + 待归因的 bounded proposal correction。",
            sty["Body"],
        )
    )
    story.append(Paragraph("与已有工作的边界", sty["H2"]))
    story.append(
        bullets(
            [
                "SeqTrack3D 已使用历史点云和历史框 sequence，不能声称首次使用序列。",
                "TrajTrack 已使用历史 bbox trajectory proposal/refinement，不能声称首次轨迹引导。",
                "HVTrack 已有固定 interval HTV，不能把普通 skip-frame 作为独立创新。",
                "你的窄边界是 within-track irregular cadence、matched time intervention 和可解释的失败/修正机制。",
            ],
            sty["Body"],
        )
    )

    # Gains
    section_start(
        story,
        "目前“涨分”的地方：哪些最强，哪些只能作为条件性信号",
        "汇报中要按证据等级呈现，不能把所有正差都并列为方法收益。",
        sty,
    )
    gain_rows = [
        ["M2 true vs matched A1", "standard", "+4.133", "+9.445", "tracklet CI 均为正", "当前最强正式 tracking 信号"],
        ["M2 true vs matched A1", "gap1124", "+2.279", "+4.143", "tracklet CI 均为正", "强 cadence 下仍为正"],
        ["R1 vs historical A1", "standard", "+4.074", "+9.318", "extra 60 epoch", "描述性，不是模块净增益"],
        ["R2 scratch M2 vs historical A1", "standard", "+2.090", "+4.640", "candidate path 不匹配", "说明 M2 可从 scratch 工作"],
        ["Scratch true vs shuffled", "random20 final", "+3.758", "+7.324", "辅助监督冲突", "partial learnability signal"],
        ["Corrected TWC C-B", "standard final", "+8.31", "+11.74", "paired control 已受损", "一致性净效应，不是端到端涨分"],
        ["Corrected TWC C-A", "standard final", "-7.00", "-12.44", "single-view 基线", "部署价值为负"],
        ["A2-order-dyn seed42 vs baseline", "standard", "约 0", "+3.35", "seed43/44 不稳定", "历史正信号，不作为当前结论"],
    ]
    story.append(
        data_table(
            ["比较", "场景", "dSuccess", "dPrecision", "限制", "证据等级"],
            gain_rows,
            [31 * mm, 22 * mm, 19 * mm, 21 * mm, 35 * mm, 37 * mm],
            sty,
            movement_cols=[2, 3],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "<b>汇报主结论应只强调前两行：</b>"
            "M2 相对 matched A1 在 standard 与 gap1124 均涨分，且 tracklet bootstrap CI 为正。"
            "其他正差用于解释研究过程，不应和正式增益混为一谈。",
            sty["Body"],
            color=GOLD,
            background=GOLD_LIGHT,
        )
    )

    # Issues and limitations
    section_start(
        story,
        "当前仍存在的问题：决定项目能否从阶段成果走向论文",
        "问题分为因果归因、训练语义、搜索可达性和外部有效性四类。",
        sty,
    )
    issue_rows = [
        ["正确时间没有优势", "standard/gap1124 的 true 不优于 fixed/shuffled", "不能使用 timestamp-aware/continuous-time headline", "冻结 physical-time No-Go"],
        ["M2 净贡献未分离", "R1 比 A1 多 60 epoch；R3 W0 塌陷", "M2-A1 不能直接写成结构因果效应", "补 A1-init W0 continuation 与 legacy W0"],
        ["proposal 语义风险", "canonical d_phys 与 candidate-frame d_box 在 nonzero candidate 不同", "两个 loss 可能对同一输出施加冲突梯度", "candidate0/1/2/3 target 与 gradient audit"],
        ["训练-递归误差过程不一致", "shared-SE(2) 可能过于干净，真实历史误差会漂移累积", "R3 collapse 可能来自 train/inference gap", "比较 A1/R1/R2 recursive error process"],
        ["pre-crop 瓶颈", "大位移时目标已离开固定 crop", "末端 innovation 无法恢复 &gt;=4 m 目标", "先做 fixed-budget predicted-history tube oracle"],
        ["统计规模不足", "单 seed、nuScenes-mini Car、mini_val 多次开发", "不能证明跨数据集/类别/seed 泛化", "full nuScenes + 第二数据集 + 3 seeds"],
        ["基线公平性", "TrajTrack 本地高分含 GT-assisted evaluator", "不能进入公平主表", "补 GT-free TrajTrack 与现代 baseline"],
        ["M3/M4 尚未验证", "当前为未提交工程稿；本地 pyquaternion 依赖缺失", "不能写成已完成贡献", "先完成 invariants、同 checkpoint arms 和停止条件"],
    ]
    story.append(
        data_table(
            ["问题", "证据", "影响", "下一步"],
            issue_rows,
            [31 * mm, 44 * mm, 44 * mm, 46 * mm],
            sty,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "<b>当前科学完成度的核心缺口不是“还没写论文”，而是“方法归因还没闭环”。</b>"
            "继续加大模型、Mamba、ODE/CDE 或 learned uncertainty 都会扩大混杂，"
            "不能替代两个 matched baseline 和 proposal 语义审计。",
            sty["Body"],
            color=ORANGE,
            background=ORANGE_LIGHT,
        )
    )

    # Next steps
    section_start(
        story,
        "下一阶段建议：先关闭归因，再决定方法论文或 benchmark/diagnosis",
        "优先级按照“最少新增混杂、最大决策价值”排序。",
        sty,
    )
    next_rows = [
        ["P0-1", "R1 same-checkpoint 2x2", "full / adapter-only / innovation-only / both-off，scale 只取 0/1", "确定运行时主要路径"],
        ["P0-2", "A1-init W0 continuation", "同 init、shared-SE(2)、60 epoch、75720 steps，关闭 M2", "排除 extra-training confound"],
        ["P0-3", "current-code legacy W0", "与 R3 同 commit/seed/steps，只恢复 legacy candidate", "解释 R3 collapse"],
        ["P0-4", "target/gradient/error-process audit", "d_obs/d_dyn/d_final 两套误差、loss cosine、递归 drift", "决定是否解锁 M1.5"],
        ["P0-5", "scratch checkpoint cross-time 2x2", "true-trained/shuffled-trained 各在 true/shuffled clock 下评测", "区分训练冲突与 inference-time 时间因果"],
        ["P1", "完整证据包", "3 seeds、full nuScenes、Waymo/KITTI-HV、GT-free baselines、效率", "满足投稿最低实验包"],
        ["条件 M1.5", "proposal 语义与误差过程修复", "d_phys/d_box 分头、相关 drift、staged ramp", "只在审计支持后启动"],
        ["条件 M3/M4", "time-agnostic path robustness / tube", "重新预注册并严格 A/B/control", "不用于复活 physical-time claim"],
    ]
    story.append(
        data_table(
            ["优先级", "工作", "固定合同", "回答的问题"],
            next_rows,
            [18 * mm, 37 * mm, 65 * mm, 45 * mm],
            sty,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        callout(
            "<b>决策分叉：</b><br/>"
            "如果 both-off/A1-init W0 保留大部分收益，转向 continuation/representation effect；<br/>"
            "如果 innovation-only 保留主要收益，围绕同语义 proposal correction 重构；<br/>"
            "如果 matched attribution 也失败，收敛为 variable-rate benchmark/diagnosis，而不是继续堆时间模块。",
            sty["Body"],
        )
    )

    # Presentation script
    section_start(
        story,
        "汇报建议：用 12 分钟讲清楚“证据如何改变创新点”",
        "这份 PDF 可作详细材料；现场口头汇报建议按以下节奏。",
        sty,
    )
    speaking_rows = [
        ["1 min", "问题与起点", "SeqTrack3D 把时间当固定帧序；真实场景存在 irregular cadence、掉帧和长 gap。"],
        ["2 min", "最初方案", "P0-P5：真实时间、Dynamics、TWC、Gate；强调工程链路已经完整。"],
        ["2 min", "第一次转向", "A1-raw/MLP/Fourier 失败，主干恢复 order-time；引出 dual-clock。"],
        ["2 min", "失败诊断", "HTV protocol dependence、crop reachability、reliability 独立 No-Go。"],
        ["2 min", "方法重构", "M0 oracle + candidate audit -> shared-SE(2) + bounded proposal innovation。"],
        ["2 min", "当前最强结果", "M2-A1 standard/gap 涨分；展示正式 controls 图，同时说明 true time 因果失败。"],
        ["1 min", "结论与下一步", "当前成果、问题、两个 matched baseline 与 2x2 attribution。"],
    ]
    story.append(
        data_table(
            ["时间", "内容", "建议表达"],
            speaking_rows,
            [19 * mm, 35 * mm, 111 * mm],
            sty,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "<b>推荐开场：</b>“这个项目最初希望把 SeqTrack3D 从固定帧序列升级到真实时间状态估计。"
            "经过多轮负对照后，我们确认 tracker 在 variable-rate 条件下可以涨分，"
            "但也发现正确 physical time 并不是当前涨分来源。今天的重点是展示："
            "我们如何通过实验逐步收窄创新点，并形成可复现的协议、诊断和 M2 方法候选。”",
            sty["Quote"],
        )
    )
    story.append(
        Paragraph(
            "<b>推荐收尾：</b>“目前我们已经把‘有涨分’和‘为什么涨分’分开。"
            "下一阶段不再增加模块，而是用同 checkpoint 2x2、matched W0 和 proposal 语义审计完成归因。"
            "归因成立就形成 bounded proposal correction 方法；归因不成立，协议与失败诊断仍可独立收敛为 benchmark 路线。”",
            sty["Quote"],
        )
    )

    # Sources
    section_start(
        story,
        "证据与可复现来源",
        "正文中的 [S#] 对应以下本地项目文件。正式数字优先以报告、CSV、checkpoint 与 hash 为准。",
        sty,
    )
    source_rows = []
    for source_id, source in SOURCES.items():
        source_rows.append(
            [
                source_id,
                source["title"],
                source["path"],
                source["role"],
            ]
        )
    story.append(
        data_table(
            ["编号", "来源", "文件", "用途"],
            source_rows,
            [14 * mm, 35 * mm, 70 * mm, 46 * mm],
            sty,
            small=True,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "<b>版本说明：</b>R1/R2/R3 与正式八组 controls 绑定 clean commit "
            "<font name='Courier'>473738f</font>。当前工作区位于后续提交之上且包含未提交的 "
            "M3/M4 工程改动，因此 M3/M4 不能与正式 M2 结果混为同一可复现实验版本。",
            sty["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "<b>报告生成说明：</b>所有自绘图均使用已复核报告中的固定数值；"
            "历史图直接复用 compare_results 中的已生成图表。source manifest "
            "同时记录图表问题、图型、结论与来源。",
            sty["BodySmall"],
        )
    )
    return story


def build_pdf(chart_paths: dict[str, Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    doc = ReportDoc(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="CT-SeqTrack 阶段性进展与成果汇报",
        author="CT-SeqTrack project report",
        subject="variable-rate 3D single object tracking progress",
    )
    story = build_story(chart_paths)
    doc.build(story, onFirstPage=cover_page, onLaterPages=later_page)


def write_manifest(chart_paths: dict[str, Path]) -> None:
    payload = {
        "report": str(PDF_PATH.relative_to(ROOT)),
        "report_date": REPORT_DATE,
        "audience": "technical",
        "delivery_mode": "pdf",
        "scope": "CT-SeqTrack project inception through 2026-07-24",
        "main_conclusion": (
            "M2 tracking signal is positive on standard and gap1124, "
            "physical-time causal claim is No-Go, and method attribution remains Hold."
        ),
        "sources": SOURCES,
        "chart_map": CHART_MAP,
        "generated_charts": {
            key: str(value.relative_to(WORKSPACE))
            for key, value in chart_paths.items()
        },
        "limitations": [
            "Current primary evidence is nuScenes-mini Car and mostly seed42.",
            "R1 relative to historical A1 contains extra training exposure.",
            "Current M3/M4 files are engineering prototypes without tracking results.",
            "The local Python environment lacks pyquaternion for the M3/M4 invariant script.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def render_pdf_for_qa() -> list[Path]:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    for old in QA_DIR.glob("page_*.png"):
        old.unlink()
    for old in QA_DIR.glob("contact_sheet_*.png"):
        old.unlink()
    document = fitz.open(PDF_PATH)
    page_paths = []
    matrix = fitz.Matrix(1.35, 1.35)
    for index, page in enumerate(document):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        page_path = QA_DIR / f"page_{index + 1:02d}.png"
        pix.save(str(page_path))
        page_paths.append(page_path)
    document.close()

    per_sheet = 12
    cols = 3
    thumb_w = 330
    gap = 18
    for sheet_index in range(math.ceil(len(page_paths) / per_sheet)):
        subset = page_paths[
            sheet_index * per_sheet : (sheet_index + 1) * per_sheet
        ]
        thumbs = []
        for page_path in subset:
            with PILImage.open(page_path) as img:
                img = img.convert("RGB")
                ratio = thumb_w / img.width
                thumb = img.resize(
                    (thumb_w, int(img.height * ratio)), PILImage.Resampling.LANCZOS
                )
                thumbs.append(thumb)
        rows = math.ceil(len(thumbs) / cols)
        thumb_h = max(img.height for img in thumbs)
        sheet = PILImage.new(
            "RGB",
            (
                cols * thumb_w + (cols + 1) * gap,
                rows * thumb_h + (rows + 1) * gap,
            ),
            "white",
        )
        for idx, thumb in enumerate(thumbs):
            col = idx % cols
            row = idx // cols
            x = gap + col * (thumb_w + gap)
            y = gap + row * (thumb_h + gap)
            sheet.paste(thumb, (x, y))
        sheet_path = QA_DIR / f"contact_sheet_{sheet_index + 1:02d}.png"
        sheet.save(sheet_path, quality=95)
    return page_paths


def verify_pdf(page_paths: list[Path]) -> dict:
    document = fitz.open(PDF_PATH)
    page_count = document.page_count
    metadata = document.metadata
    text_pages = []
    for index in range(page_count):
        text = document[index].get_text("text").strip()
        text_pages.append(
            {
                "page": index + 1,
                "characters": len(text),
                "has_text": bool(text),
            }
        )
    document.close()
    if page_count < 15:
        raise RuntimeError(f"Report is unexpectedly short: {page_count} pages")
    if len(page_paths) != page_count:
        raise RuntimeError("Rendered QA page count does not match PDF page count")
    if any(not item["has_text"] for item in text_pages[1:]):
        raise RuntimeError("One or more non-cover pages have no extractable text")
    return {
        "pdf": str(PDF_PATH),
        "bytes": PDF_PATH.stat().st_size,
        "page_count": page_count,
        "metadata": metadata,
        "text_pages": text_pages,
        "qa_contact_sheets": [
            str(path)
            for path in sorted(QA_DIR.glob("contact_sheet_*.png"))
        ],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths = generate_charts()
    build_pdf(chart_paths)
    write_manifest(chart_paths)
    page_paths = render_pdf_for_qa()
    verification = verify_pdf(page_paths)
    verification_path = QA_DIR / "verification.json"
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
