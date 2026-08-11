from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .data_loader import Config, TARGET, build_dataset, make_stratified_split, require_files
    from .eval_metrics import PRIMARY_METRICS
    from .model_baseline import (
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        train_and_save_baseline,
    )
except ImportError:  # direct script compatibility
    from data_loader import Config, TARGET, build_dataset, make_stratified_split, require_files
    from eval_metrics import PRIMARY_METRICS
    from model_baseline import (
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        train_and_save_baseline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and train the YadYar Lite Phase 1 baseline."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    config = Config()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    require_files(data_dir)
    dataset = build_dataset(data_dir, config)
    dataset.to_csv(output_dir / "phase1_engineered_dataset.csv", index=False)

    y = dataset[TARGET]
    train_idx, test_idx = make_stratified_split(dataset, config)

    train_and_save_baseline(
        dataset,
        train_idx,
        model_dir / "phase1_logistic_baseline.joblib",
        config,
    )

    target_counts = y.value_counts().sort_index()
    summary = {
        "scope": f"{config.module}/{config.presentation}",
        "students": int(len(dataset)),
        "not_at_risk": int(target_counts.get(0, 0)),
        "at_risk": int(target_counts.get(1, 0)),
        "at_risk_rate": float(y.mean()),
        "train_rows": int(len(train_idx)),
        "test_rows_reserved_for_phase2": int(len(test_idx)),
        "raw_feature_count": len(FEATURES),
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_definition": {"1": "Fail or Withdrawn", "0": "Pass or Distinction"},
        "time_window": "course days 0 through 30",
        "baseline": "scikit-learn LogisticRegression with preprocessing pipeline",
        "planned_primary_metrics": PRIMARY_METRICS,
        "note": "Phase 1 validates data preparation and baseline training; final held-out evaluation is reported in Phase 2.",
    }
    (output_dir / "phase1_setup_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    split_table = pd.DataFrame(
        [
            {
                "split": "train",
                "n_students": len(train_idx),
                "at_risk": int(y.loc[train_idx].sum()),
                "not_at_risk": int((1 - y.loc[train_idx]).sum()),
                "at_risk_rate": float(y.loc[train_idx].mean()),
            },
            {
                "split": "test_reserved_for_phase2",
                "n_students": len(test_idx),
                "at_risk": int(y.loc[test_idx].sum()),
                "not_at_risk": int((1 - y.loc[test_idx]).sum()),
                "at_risk_rate": float(y.loc[test_idx].mean()),
            },
        ]
    )
    split_table.to_csv(output_dir / "phase1_split_plan.csv", index=False)

    print("Phase 1 baseline setup completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
