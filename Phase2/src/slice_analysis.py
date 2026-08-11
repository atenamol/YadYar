from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

try:
    from .data_loader import TARGET
except ImportError:
    from data_loader import TARGET


def engagement_group(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first")
    try:
        return pd.qcut(ranks, q=3, labels=["Low", "Medium", "High"])
    except ValueError:
        return pd.Series("All", index=series.index, dtype="object")


def engagement_breakdown(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in analysis.groupby("engagement_group", observed=True):
        rows.append(
            {
                "engagement_group": str(group_name),
                "n_students": len(group),
                "at_risk_rate": group[TARGET].mean(),
                "recall_at_risk": recall_score(group[TARGET], group["prediction"], zero_division=0),
                "precision_at_risk": precision_score(group[TARGET], group["prediction"], zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def save_engagement_breakdown(analysis: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    result = engagement_breakdown(analysis)
    result.to_csv(output_dir / "engagement_group_analysis.csv", index=False)
    return result


def early_assessment_slice(test: pd.DataFrame) -> pd.DataFrame:
    work = test.copy()
    work["early_assessment_status"] = np.where(
        work["assessment_count_30d"] > 0,
        "Submitted early assessment",
        "No early assessment",
    )
    rows = []
    for group, part in work.groupby("early_assessment_status"):
        y = part[TARGET].to_numpy()
        pred = part["prediction"].to_numpy()
        rows.append({
            "slice": group,
            "students": len(part),
            "at_risk_students": int(y.sum()),
            "at_risk_rate": float(y.mean()) if len(y) else 0.0,
            "precision_at_risk": precision_score(y, pred, zero_division=0),
            "recall_at_risk": recall_score(y, pred, zero_division=0),
            "false_negatives": int(((y == 1) & (pred == 0)).sum()),
            "false_positives": int(((y == 0) & (pred == 1)).sum()),
        })
    return pd.DataFrame(rows)


def save_early_assessment_slice(test: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    result = early_assessment_slice(test)
    result.to_csv(output_dir / "bonus_early_assessment_slice.csv", index=False)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    plotted = result.set_index("slice")[["precision_at_risk", "recall_at_risk"]]
    plotted.plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Bonus slice: performance by early-assessment availability")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    figure.savefig(output_dir / "bonus_early_assessment_slice.png", dpi=180)
    plt.close(figure)
    return result
