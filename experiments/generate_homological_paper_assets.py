#!/usr/bin/env python3
"""Generate LaTeX tables, copied figures, and a manifest for the homological paper."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from benchmark_common import DOMAIN_DISPLAY, MODEL_DISPLAY


DOMAIN_ORDER = ["square", "circle", "stadium_w0p6", "stadium_w1p2"]
MODEL_ORDER = [
    "billiard",
    "brownian_low",
    "brownian_med",
    "brownian_high",
    "vicsek_low",
    "vicsek_med",
    "vicsek_high",
    "homological",
    "sequential_homological",
]


def parse_run_mapping(entries: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected DOMAIN=PATH entry, got: {entry}")
        domain, raw_path = entry.split("=", 1)
        domain = domain.strip()
        path = Path(raw_path.strip())
        if domain not in DOMAIN_DISPLAY:
            raise ValueError(f"Unsupported domain key: {domain}")
        if not path.exists():
            raise FileNotFoundError(f"Run directory not found: {path}")
        mapping[domain] = path
    return mapping


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_pm(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.2f} $\\pm$ {std_val:.2f}"


def bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def red(text: str) -> str:
    return f"\\textcolor{{red!70!black}}{{{text}}}"


def blue(text: str) -> str:
    return f"\\textcolor{{blue!70!black}}{{{text}}}"


def write_square_config_table(agg_df: pd.DataFrame, output_path: Path) -> None:
    square_df = agg_df[agg_df["domain"] == "square"].copy()
    present_models = [model for model in MODEL_ORDER if model in set(square_df["model"].astype(str))]
    display_names = {
        "billiard": "Billiard",
        "brownian_low": "Brownian-L",
        "brownian_med": "Brownian-M",
        "brownian_high": "Brownian-H",
        "vicsek_low": "Vicsek-L",
        "vicsek_med": "Vicsek-M",
        "vicsek_high": "Vicsek-H",
        "homological": "Homological",
        "sequential_homological": "Sequential",
    }
    column_spec = "cc" + ("r" * len(present_models))
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{3.5pt}",
        "    \\caption{Unit-square capped detection time over the $(n,r)$ grid. In each row, the best mean is shown in red, the second-best in blue, and the third-best in bold. Entries shown as $\\--$ correspond to cells that remained at the cap $T=5.0\\,\\mathrm{s}$ in all held-out runs, i.e. they did not achieve detection within the allotted time.}",
        "    \\label{tab:config-results}",
        "    \\resizebox{\\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{column_spec}}}",
        "        \\toprule",
        "        $n$ & $r$ & " + " & ".join(display_names[m] for m in present_models) + " \\\\",
        "        \\midrule",
    ]
    for n_val in sorted(square_df["n"].unique()):
        for r_val in sorted(square_df["r"].unique()):
            row_df = square_df[(square_df["n"] == n_val) & (square_df["r"] == r_val)].copy()
            if row_df.empty:
                continue
            entries = [str(int(n_val)), f"{float(r_val):.2f}"]
            ranked_models = sorted(
                [
                    (
                        str(row["model"]),
                        float(row["mean_tau"]),
                        float(row["fail_rate"]),
                    )
                    for _, row in row_df.iterrows()
                ],
                key=lambda item: (item[1], item[2], MODEL_ORDER.index(item[0])),
            )
            rank_style = {}
            if len(ranked_models) >= 1:
                rank_style[ranked_models[0][0]] = red
            if len(ranked_models) >= 2:
                rank_style[ranked_models[1][0]] = blue
            if len(ranked_models) >= 3:
                rank_style[ranked_models[2][0]] = bold
            for model_name in present_models:
                model_row = row_df[row_df["model"] == model_name]
                if model_row.empty:
                    entries.append("--")
                    continue
                mean_val = float(model_row.iloc[0]["mean_tau"])
                std_val = float(model_row.iloc[0]["std_tau"])
                if abs(mean_val - 5.0) <= 1e-12 and abs(std_val) <= 1e-12:
                    text = "$\\--$"
                else:
                    text = fmt_pm(mean_val, std_val)
                style = rank_style.get(model_name)
                if style is not None:
                    text = style(text)
                entries.append(text)
            lines.append("        " + " & ".join(entries) + " \\\\")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "\\end{table*}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_square_model_summary_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    square_df = summary_df[summary_df["domain"] == "square"].copy()
    square_df["model"] = pd.Categorical(square_df["model"], categories=MODEL_ORDER, ordered=True)
    square_df = square_df.sort_values("model")
    best_model = None
    if not square_df.empty:
        best_model = str(square_df.loc[square_df["grid_mean_tau"].idxmin(), "model"])
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Unit-square model-level summary over the 20-cell grid.}",
        "    \\label{tab:model-summary}",
        "    \\begin{tabular}{lcccc}",
        "        \\toprule",
        "        Model & Grid mean $\\tau$ & Mean success $\\tau$ & Fail rate & Best cells \\\\",
        "        \\midrule",
    ]
    for _, row in square_df.iterrows():
        highlight = str(row["model"]) == best_model
        values = [
            str(row["model_display"]),
            f"{float(row['grid_mean_tau']):.2f}",
            f"{float(row['grid_mean_success_tau']):.2f}",
            f"{float(row['grid_fail_rate']):.2f}",
            str(int(row["best_cell_count"])),
        ]
        if highlight:
            values = [bold(v) for v in values]
        lines.append(
            "        "
            + " & ".join(values)
            + " \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_cross_domain_summary_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df = summary_df.copy()
    summary_df["domain"] = pd.Categorical(summary_df["domain"], categories=DOMAIN_ORDER, ordered=True)
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["domain", "model"])
    best_by_domain = (
        summary_df.loc[summary_df.groupby("domain", observed=True)["grid_mean_tau"].idxmin(), ["domain", "model"]]
        .set_index("domain")["model"]
        .to_dict()
    )
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\caption{Cross-domain model summary. Each entry averages over the 20-cell $(n,r)$ grid within that domain.}",
        "    \\label{tab:cross-domain-summary}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{l l c c c c}",
        "        \\toprule",
        "        Domain & Model & Grid mean $\\tau$ & Mean success $\\tau$ & Fail rate & Best cells \\\\",
        "        \\midrule",
    ]
    for _, row in summary_df.iterrows():
        highlight = best_by_domain.get(row["domain"]) == row["model"]
        values = [
            DOMAIN_DISPLAY.get(str(row["domain"]), str(row["domain"])),
            str(row["model_display"]),
            f"{float(row['grid_mean_tau']):.2f}",
            f"{float(row['grid_mean_success_tau']):.2f}",
            f"{float(row['grid_fail_rate']):.2f}",
            str(int(row["best_cell_count"])),
        ]
        if highlight:
            values = [bold(v) for v in values]
        lines.append(
            "        "
            + " & ".join(values)
            + " \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "\\end{table*}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_paired_overall_table(paired_df: pd.DataFrame, output_path: Path) -> None:
    paired_df = paired_df.copy()
    comp_labels = {
        "homological_minus_billiard": "Homological - Billiard",
        "homological_minus_brownian_low": "Homological - Brownian (Low)",
        "homological_minus_brownian_med": "Homological - Brownian (Medium)",
        "homological_minus_brownian_high": "Homological - Brownian (High)",
        "homological_minus_vicsek_low": "Homological - Vicsek (Low)",
        "homological_minus_vicsek_med": "Homological - Vicsek (Medium)",
        "homological_minus_vicsek_high": "Homological - Vicsek (High)",
        "sequential_homological_minus_billiard": "Sequential Homological - Billiard",
        "sequential_homological_minus_brownian_high": "Sequential Homological - Brownian (High)",
        "sequential_homological_minus_homological": "Sequential Homological - Homological",
    }
    paired_df["domain"] = pd.Categorical(paired_df["domain"], categories=DOMAIN_ORDER, ordered=True)
    paired_df = paired_df.sort_values(["domain", "comparison"])
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\caption{Paired matched-run capped-time deltas. Negative values favor Homological Motion.}",
        "    \\label{tab:paired-overall}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{l l c c c}",
        "        \\toprule",
        "        Domain & Comparison & Mean $\\Delta\\tau$ & 95\\% bootstrap CI & Paired success count \\\\",
        "        \\midrule",
    ]
    for _, row in paired_df.iterrows():
        ci = f"[{float(row['ci_low']):.2f}, {float(row['ci_high']):.2f}]"
        lines.append(
            "        "
            + " & ".join(
                [
                    DOMAIN_DISPLAY.get(str(row["domain"]), str(row["domain"])),
                    comp_labels.get(str(row["comparison"]), str(row["comparison"])),
                    f"{float(row['mean_delta_tau']):.2f}",
                    ci,
                    str(int(row["paired_success_count"])),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "\\end{table*}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def copy_run_figures(run_dir: Path, figure_dir: Path) -> List[Dict[str, str]]:
    copied: List[Dict[str, str]] = []
    heatmap_dir = run_dir / "heatmaps"
    if not heatmap_dir.exists():
        return copied
    for src in sorted(heatmap_dir.glob("*.png")):
        dst = figure_dir / f"paper_{src.name}"
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "target": str(dst)})
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX assets for the homological-dynamics paper.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Benchmark run mapping in the form domain=/abs/or/relative/path/to/run_dir. Repeat per domain.",
    )
    parser.add_argument("--paper-dir", type=str, default="latex/homologicalDynamics")
    args = parser.parse_args()

    run_dirs = parse_run_mapping(args.run)
    paper_dir = Path(args.paper_dir)
    figure_dir = paper_dir / "figures"
    table_dir = paper_dir / "tables"
    ensure_dir(figure_dir)
    ensure_dir(table_dir)

    aggregate_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    paired_frames: List[pd.DataFrame] = []
    copied_figures: List[Dict[str, str]] = []
    manifest_runs: Dict[str, Dict[str, str]] = {}

    for domain in DOMAIN_ORDER:
        if domain not in run_dirs:
            continue
        run_dir = run_dirs[domain]
        agg_path = run_dir / "aggregate_stats.csv"
        summary_path = run_dir / "model_summary.csv"
        paired_path = run_dir / "paired_overall_summary.csv"
        if agg_path.exists():
            aggregate_frames.append(pd.read_csv(agg_path))
        if summary_path.exists():
            summary_frames.append(pd.read_csv(summary_path))
        if paired_path.exists():
            paired_frames.append(pd.read_csv(paired_path))
        copied_figures.extend(copy_run_figures(run_dir, figure_dir))
        manifest_runs[domain] = {
            "run_dir": str(run_dir),
            "aggregate_stats_csv": str(agg_path),
            "model_summary_csv": str(summary_path),
            "paired_overall_summary_csv": str(paired_path),
        }

    if not aggregate_frames or not summary_frames:
        raise ValueError("No valid benchmark outputs were found.")

    agg_df = pd.concat(aggregate_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    paired_df = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()

    if "square" in run_dirs:
        write_square_config_table(agg_df, table_dir / "config_results_table_generated.tex")
        write_square_model_summary_table(summary_df, table_dir / "model_summary_table_generated.tex")
    write_cross_domain_summary_table(summary_df, table_dir / "cross_domain_summary_table_generated.tex")
    if not paired_df.empty:
        write_paired_overall_table(paired_df, table_dir / "paired_overall_table_generated.tex")

    manifest = {
        "script": "experiments/generate_homological_paper_assets.py",
        "paper_dir": str(paper_dir),
        "runs": manifest_runs,
        "tables": {
            "config_results_table_generated": str(table_dir / "config_results_table_generated.tex"),
            "model_summary_table_generated": str(table_dir / "model_summary_table_generated.tex"),
            "cross_domain_summary_table_generated": str(table_dir / "cross_domain_summary_table_generated.tex"),
            "paired_overall_table_generated": str(table_dir / "paired_overall_table_generated.tex"),
        },
        "copied_figures": copied_figures,
    }
    (paper_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
