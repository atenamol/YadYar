from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loader import KEYS, NUMERIC_FEATURES, TARGET
    from .slice_analysis import engagement_group, save_engagement_breakdown
except ImportError:
    from data_loader import KEYS, NUMERIC_FEATURES, TARGET
    from slice_analysis import engagement_group, save_engagement_breakdown


def save_error_analysis(
    test_rows: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    analysis = test_rows.copy()
    analysis["risk_probability"] = probability
    analysis["prediction"] = (probability >= threshold).astype(int)
    analysis["error_type"] = "Correct"
    analysis.loc[(analysis[TARGET] == 1) & (analysis["prediction"] == 0), "error_type"] = "False Negative"
    analysis.loc[(analysis[TARGET] == 0) & (analysis["prediction"] == 1), "error_type"] = "False Positive"
    analysis["engagement_group"] = engagement_group(analysis["total_clicks_30d"])

    visible_columns = KEYS + [
        "final_result", TARGET, "prediction", "risk_probability", "error_type", "engagement_group",
    ] + NUMERIC_FEATURES + ["age_band", "highest_education", "disability"]

    errors = analysis[analysis["error_type"] != "Correct"].copy()
    errors.sort_values(["error_type", "risk_probability"], inplace=True)
    errors[visible_columns].to_csv(output_dir / "all_errors.csv", index=False)

    false_negatives = errors[errors["error_type"] == "False Negative"].sort_values("risk_probability", ascending=False)
    false_positives = errors[errors["error_type"] == "False Positive"].sort_values("risk_probability", ascending=False)
    false_negatives[visible_columns].head(20).to_csv(output_dir / "representative_false_negatives.csv", index=False)
    false_positives[visible_columns].head(20).to_csv(output_dir / "representative_false_positives.csv", index=False)

    save_engagement_breakdown(analysis, output_dir)
    return analysis


def write_manual_summary(
    analysis: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    threshold: float,
) -> None:
    fn = analysis[analysis["error_type"] == "False Negative"]
    fp = analysis[analysis["error_type"] == "False Positive"]
    best = metrics.sort_values(["recall_at_risk", "total_cost"], ascending=[False, True]).iloc[0]
    text = f"""# Manual Error Summary

## Operating point

The main decision threshold is **{threshold:.2f}**. A student is flagged as at risk when the predicted probability is at least this value.

## Representative failure patterns

- **False negatives: {len(fn)} students.** These are the most serious errors because an at-risk student would not receive an intervention. Inspect `representative_false_negatives.csv`, especially cases with moderate or high early activity but a later `Fail` or `Withdrawn` result.
- **False positives: {len(fp)} students.** These students would receive unnecessary support. Inspect `representative_false_positives.csv`, especially students with low first-30-day activity who later passed.
- Compare `engagement_group_analysis.csv` to determine whether recall is weaker for low-, medium-, or high-engagement students.

## Best observed row in the generated comparison

- Model: **{best['model']}**
- Threshold: **{best['threshold']:.2f}**
- At-risk recall: **{best['recall_at_risk']:.3f}**
- At-risk precision: **{best['precision_at_risk']:.3f}**
- PR-AUC: **{best['pr_auc']:.3f}**
- Cost: **{int(best['total_cost'])}**, where FN costs 5 and FP costs 1.

## Interpretation guidance

Do not claim that a flagged student will certainly fail. The output is an early-warning probability intended to prioritize human support. Discuss the trade-off: lowering the threshold generally increases recall and false positives, while raising it reduces alerts but may miss more at-risk students.
"""
    (output_dir / "manual_error_summary.md").write_text(text, encoding="utf-8")


def prediction_contract() -> dict:
    return {
        "input": {
            "description": "One student profile using only information available through day 30.",
            "required_feature_count": 16,
        },
        "output": {
            "risk_probability": "float in [0, 1]",
            "at_risk_flag": "boolean based on the configured threshold",
            "recommended_action": "supportive_human_review or no_flag_at_current_threshold",
        },
        "safety_note": "The flag supports human review and is not an automatic academic decision.",
    }
