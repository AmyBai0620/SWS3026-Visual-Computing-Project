"""Three-model landmark benchmark comparison (v1).

Reads the reviewed 350-sample metrics of MediaPipe Face Mesh, RTMPose-Face and
FAN, normalizes them into one table, and writes comparison figures plus a
markdown report to outputs/comparison/.

Usage:
    python compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "v1"

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "comparison"

MODELS = ["MediaPipe", "RTMPose-Face", "FAN"]
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Categorical slots 1-3 (validated for light surface; magenta is sub-3:1 so
# every bar carries a direct value label and the report repeats the numbers
# in tables).
MODEL_COLORS = {
    "MediaPipe": "#2a78d6",
    "RTMPose-Face": "#008300",
    "FAN": "#e87ba4",
}
# Status palette (fixed): good / warning / critical.
REVIEW_COLORS = {"good": "#0ca30c", "acceptable": "#fab219", "wrong": "#d03b3b"}
REVIEW_LABELS_ZH = {"good": "Good", "acceptable": "Acceptable", "wrong": "Wrong"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_mediapipe() -> pd.DataFrame:
    csv = BASE_DIR / "outputs" / "mediapipe" / "sample_350" / "sample_metrics_reviewed.csv"
    df = pd.read_csv(csv)
    return pd.DataFrame(
        {
            "model": "MediaPipe",
            "sample_id": df["sample_id"],
            "class_name": df["label"],
            "runtime_ms": df["runtime_ms"],
            "review_label": df["landmark_valid"].str.lower(),
            "success": df["extract_success"].astype(str).str.lower() == "true",
            "score_mean": np.nan,
        }
    )


def load_topdown(name: str, folder: str) -> pd.DataFrame:
    csv = BASE_DIR / "outputs" / folder / "sample_350" / "sample_metrics_reviewed.csv"
    df = pd.read_csv(csv)
    return pd.DataFrame(
        {
            "model": name,
            "sample_id": df["sample_id"],
            "class_name": df["class_name"],
            "runtime_ms": df["inference_ms"],
            "review_label": df["review_label"].str.lower(),
            "success": df["inference_success"].astype(str).str.lower() == "true",
            "score_mean": df["score_mean"],
        }
    )


def overall_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        d = data[data["model"] == model]
        n = len(d)
        counts = d["review_label"].value_counts()
        good = int(counts.get("good", 0))
        acceptable = int(counts.get("acceptable", 0))
        wrong = int(counts.get("wrong", 0))
        rows.append(
            {
                "model": model,
                "samples": n,
                "raw_success_rate": d["success"].mean(),
                "good": good,
                "acceptable": acceptable,
                "wrong": wrong,
                "manual_valid_rate": (good + acceptable) / n,
                "strict_good_rate": good / n,
                "mean_ms": d["runtime_ms"].mean(),
                "median_ms": d["runtime_ms"].median(),
                "p95_ms": d["runtime_ms"].quantile(0.95),
                "mean_score": d["score_mean"].mean(),
            }
        )
    return pd.DataFrame(rows)


def per_class_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for cls in CLASSES:
            d = data[(data["model"] == model) & (data["class_name"] == cls)]
            n = len(d)
            good = int((d["review_label"] == "good").sum())
            wrong = int((d["review_label"] == "wrong").sum())
            rows.append(
                {
                    "model": model,
                    "class_name": cls,
                    "samples": n,
                    "good": good,
                    "wrong": wrong,
                    "good_rate": good / n,
                    "manual_valid_rate": (n - wrong) / n,
                }
            )
    return pd.DataFrame(rows)


def style_axes(ax, value_axis: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis=value_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def fig_review_outcome(overall: pd.DataFrame) -> str:
    fig, ax = new_fig(9, 3.4)
    y = np.arange(len(MODELS))[::-1]
    left = np.zeros(len(MODELS))
    for label in ("good", "acceptable", "wrong"):
        vals = overall.set_index("model").loc[MODELS, label].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.52, color=REVIEW_COLORS[label],
                edgecolor=SURFACE, linewidth=2, label=REVIEW_LABELS_ZH[label])
        for yi, v, l in zip(y, vals, left):
            if v >= 14:
                ax.text(l + v / 2, yi, f"{int(v)}", ha="center", va="center",
                        fontsize=10, color="#ffffff", fontweight="bold")
            elif v > 0:
                ax.text(l + v + 4, yi + 0.32, f"{int(v)}", ha="left", va="center",
                        fontsize=9, color=INK_2)
        left += vals
    ax.set_yticks(y, MODELS)
    ax.tick_params(axis="y", labelsize=11, labelcolor=INK)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Samples (of 350)", color=INK_2, fontsize=10)
    ax.set_title("Manual review outcome per model (350 samples)",
                 color=INK, fontsize=13, loc="left", pad=14)
    style_axes(ax, "x")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2,
              ncols=3, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    name = "fig1_review_outcome.png"
    fig.savefig(OUT_DIR / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return name


def grouped_bar(per_class: pd.DataFrame, value_col: str, title: str,
                ylabel: str, name: str, ymax: float = 100.0) -> str:
    fig, ax = new_fig(10.5, 4.2)
    x = np.arange(len(CLASSES))
    width = 0.26
    for i, model in enumerate(MODELS):
        d = per_class[per_class["model"] == model].set_index("class_name")
        vals = d.loc[CLASSES, value_col].to_numpy(dtype=float) * 100
        pos = x + (i - 1) * (width + 0.02)
        ax.bar(pos, vals, width=width, color=MODEL_COLORS[model], label=model)
        for p, v in zip(pos, vals):
            ax.text(p, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(x, [c.capitalize() for c in CLASSES])
    ax.tick_params(axis="x", labelsize=10.5, labelcolor=INK)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=14)
    style_axes(ax, "y")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2,
              ncols=3, bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_runtime(overall: pd.DataFrame) -> str:
    fig, ax = new_fig(9, 3.2)
    d = overall.set_index("model").loc[MODELS]
    y = np.arange(len(MODELS))[::-1]
    vals = d["mean_ms"].to_numpy(dtype=float)
    ax.barh(y, vals, height=0.52, color=[MODEL_COLORS[m] for m in MODELS])
    for yi, v, p95 in zip(y, vals, d["p95_ms"]):
        ax.text(v + 3, yi, f"{v:.1f} ms  (P95 {p95:.1f})", ha="left",
                va="center", fontsize=10, color=INK_2)
    ax.set_yticks(y, MODELS)
    ax.tick_params(axis="y", labelsize=11, labelcolor=INK)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("Mean inference time per image (ms, CPU)", color=INK_2, fontsize=10)
    ax.set_title("Inference speed (lower is better)", color=INK, fontsize=13,
                 loc="left", pad=14)
    style_axes(ax, "x")
    fig.tight_layout()
    name = "fig4_runtime.png"
    fig.savefig(OUT_DIR / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return name


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_report(overall: pd.DataFrame, per_class: pd.DataFrame,
                 figures: dict[str, str]) -> Path:
    o = overall.set_index("model")
    lines: list[str] = []
    add = lines.append
    add(f"# 三模型关键点基准对比报告 ({VERSION})")
    add("")
    add("- 数据：FER-2013 train，7 类 × 50 张 = 350 张（`manifests/sample_manifest_350.csv`）")
    add("- 模型：MediaPipe Face Mesh 0.10.21（468 点，自带检测器）、"
        "RTMPose-M Face6 256x256（106 点）、2D FAN-4（68 点），均为 CPU 推理")
    add("- 本报告由 `compare_models.py` 从三份 `sample_metrics_reviewed.csv` 自动生成")
    add("")
    add("## 1. 总体对比")
    add("")
    add("| 指标 | MediaPipe | RTMPose-Face | FAN |")
    add("|---|---:|---:|---:|")

    def row(label: str, fn) -> None:
        add(f"| {label} | " + " | ".join(fn(o.loc[m]) for m in MODELS) + " |")

    row("原始提取成功率", lambda r: fmt_pct(r["raw_success_rate"]))
    row("人工有效率 (good+acceptable)", lambda r: fmt_pct(r["manual_valid_rate"]))
    row("严格 good 率", lambda r: fmt_pct(r["strict_good_rate"]))
    row("Good / Acceptable / Wrong",
        lambda r: f"{int(r['good'])} / {int(r['acceptable'])} / {int(r['wrong'])}")
    row("平均耗时", lambda r: f"{r['mean_ms']:.2f} ms")
    row("中位数耗时", lambda r: f"{r['median_ms']:.2f} ms")
    row("P95 耗时", lambda r: f"{r['p95_ms']:.2f} ms")
    row("平均关键点置信度",
        lambda r: "—" if pd.isna(r["mean_score"]) else f"{r['mean_score']:.3f}")
    add("")
    add(f"![人工复核结果]({figures['review']})")
    add("")
    add("**口径说明（重要）**：RTMPose-Face 和 FAN 将整张图作为外部人脸框输入，被强制输出全部"
        "关键点，因此“原始成功率 100%”不代表全部对齐正确；MediaPipe 自带人脸检测器，其失败"
        "（33 张）为显式的“检测不到脸”。跨模型比较应以**人工有效率**与**严格 good 率**为准。")
    add("")
    add("## 2. 分类别严格 good 率")
    add("")
    add(f"![分类别 good 率]({figures['good']})")
    add("")
    add("| 类别 | MediaPipe | RTMPose-Face | FAN |")
    add("|---|---:|---:|---:|")
    pc = per_class.set_index(["model", "class_name"])
    for cls in CLASSES:
        cells = " | ".join(fmt_pct(pc.loc[(m, cls), "good_rate"]) for m in MODELS)
        add(f"| {cls.capitalize()} | {cells} |")
    add("")
    add("## 3. 分类别人工有效率 (good+acceptable)")
    add("")
    add(f"![分类别有效率]({figures['valid']})")
    add("")
    add("| 类别 | MediaPipe | RTMPose-Face | FAN |")
    add("|---|---:|---:|---:|")
    for cls in CLASSES:
        cells = " | ".join(fmt_pct(pc.loc[(m, cls), "manual_valid_rate"]) for m in MODELS)
        add(f"| {cls.capitalize()} | {cells} |")
    add("")
    add("## 4. 推理速度")
    add("")
    add(f"![推理速度]({figures['runtime']})")
    add("")
    mp, fan = o.loc["MediaPipe"], o.loc["FAN"]
    rtm = o.loc["RTMPose-Face"]
    add(f"MediaPipe 平均 {mp['mean_ms']:.2f} ms/张，约为 RTMPose-Face 的 "
        f"{rtm['mean_ms'] / mp['mean_ms']:.0f} 倍、FAN 的 "
        f"{fan['mean_ms'] / mp['mean_ms']:.0f} 倍速度。按 FER-2013 全量约 3.5 万张估算："
        f"MediaPipe 约 {35000 * mp['mean_ms'] / 60000:.0f} 分钟，RTMPose-Face 约 "
        f"{35000 * rtm['mean_ms'] / 3600000:.1f} 小时，FAN 约 "
        f"{35000 * fan['mean_ms'] / 3600000:.1f} 小时（CPU）。")
    add("")
    add("## 5. 主要发现")
    add("")
    add("1. **质量维度 FAN 最好**：严格 good 率最高"
        f"（{fmt_pct(fan['strict_good_rate'])}），关键点置信度最高"
        f"（{fan['mean_score']:.3f}），但有 {int(fan['wrong'])} 张 wrong"
        "（集中在大侧脸、手/头发遮挡、严重模糊），且速度最慢。")
    add(f"2. **鲁棒性 RTMPose-Face 最好**：0 张 wrong，人工有效率 100%，"
        "但 good 率最低，大量样本仅达 acceptable。")
    add(f"3. **速度 MediaPipe 碾压**：{mp['mean_ms']:.2f} ms/张，唯一可实时的方案；"
        f"代价是 {int(mp['wrong'])} 张（{fmt_pct(1 - mp['manual_valid_rate'])}）检测失败。")
    add("4. **难度分布三模型一致**：Sad、Angry、Disgust 的 good 率在三个模型上都偏低，"
        "Happy 都最高，说明差异来自各类困难样本（侧脸、遮挡、暗光、低分辨率）比例，"
        "而非模型的表情偏好。")
    add("")
    add("## 6. v1 结论与建议")
    add("")
    add("- 建议 **MediaPipe + FAN** 进入全量：MediaPipe 提供实时能力与 468 点信息量，"
        "FAN 作为质量上限参照；RTMPose-Face 质量未超过 FAN、速度远不及 MediaPipe，"
        "仅在追求零失败时作为 FAN 的替代。")
    add("- MediaPipe 的 33 张失败样本按原计划单独做 padding / CLAHE / 降阈值补救实验，"
        "结果与默认配置分开记录。")
    add("- 补充实验建议：给 FAN / RTMPose-Face 前置真实人脸检测框（而非整图），"
        "观察贴合质量是否提升。")
    add("- 最终取舍以同一分类器下的 Accuracy / Macro F1 为准。")
    add("")
    report = OUT_DIR / f"comparison_report_{VERSION}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Segoe UI", "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    data = pd.concat(
        [
            load_mediapipe(),
            load_topdown("RTMPose-Face", "rtmpose_face"),
            load_topdown("FAN", "fan_face"),
        ],
        ignore_index=True,
    )
    data.to_csv(OUT_DIR / f"combined_metrics_{VERSION}.csv", index=False)

    overall = overall_table(data)
    per_class = per_class_table(data)
    overall.to_csv(OUT_DIR / f"overall_summary_{VERSION}.csv", index=False)
    per_class.to_csv(OUT_DIR / f"per_class_summary_{VERSION}.csv", index=False)

    figures = {
        "review": fig_review_outcome(overall),
        "good": grouped_bar(per_class, "good_rate",
                            "Strict good rate per class (%)", "Good rate (%)",
                            "fig2_per_class_good_rate.png", ymax=80),
        "valid": grouped_bar(per_class, "manual_valid_rate",
                             "Manual valid rate per class (good + acceptable, %)",
                             "Valid rate (%)", "fig3_per_class_valid_rate.png"),
        "runtime": fig_runtime(overall),
    }
    report = write_report(overall, per_class, figures)
    print(f"Report: {report}")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
