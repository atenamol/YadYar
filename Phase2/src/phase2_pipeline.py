from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .data_loader import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, build_dataset, require_files)
    from .error_inspector import prediction_contract, save_error_analysis, write_manual_summary
    from .model_evaluation import (metrics_at_threshold, model_factories, positive_probability, save_calibration_plot, save_confusion_matrix, save_lightgbm_feature_importance, save_operating_point_plots, save_pr_curve)
except ImportError:
    from data_loader import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, build_dataset, require_files)
    from error_inspector import prediction_contract, save_error_analysis, write_manual_summary
    from model_evaluation import (metrics_at_threshold, model_factories, positive_probability, save_calibration_plot, save_confusion_matrix, save_lightgbm_feature_importance, save_operating_point_plots, save_pr_curve)


@dataclass(frozen=True)
class Config:
    start_day: int = 0
    end_day: int = 30
    test_size: float = 0.20
    random_state: int = 42
    main_threshold: float = 0.30
    fn_cost: int = 5
    fp_cost: int = 1
    chunksize: int = 500_000
    module: str | None = None
    presentation: str | None = None


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    model_dir: Path,
    config: Config,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    require_files(data_dir)

    dataset = build_dataset(data_dir, config)
    dataset.to_csv(output_dir / "engineered_student_dataset.csv", index=False)

    class_counts = dataset[TARGET].value_counts().sort_index()
    class_summary = {
        "not_at_risk": int(class_counts.get(0, 0)),
        "at_risk": int(class_counts.get(1, 0)),
        "at_risk_rate": float(dataset[TARGET].mean()),
        "rows": int(len(dataset)),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(class_summary, indent=2), encoding="utf-8")

    x = dataset[FEATURES]
    y = dataset[TARGET]
    train_indices, test_indices = train_test_split(
        dataset.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    x_train, x_test = x.loc[train_indices], x.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    rows: list[dict[str, Any]] = []
    probabilities = {}
    trained_models = {}
    for model_name, model in model_factories(config, y_train).items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        probability = positive_probability(model, x_test)
        probabilities[model_name] = probability
        trained_models[model_name] = model
        for threshold in [0.50, config.main_threshold]:
            result = metrics_at_threshold(y_test, probability, threshold, config)
            result["model"] = model_name
            rows.append(result)
            save_confusion_matrix(y_test, probability, threshold, model_name, output_dir)

    metrics = pd.DataFrame(rows)[[
        "model", "threshold", "accuracy", "precision_at_risk", "recall_at_risk",
        "f1_at_risk", "pr_auc", "roc_auc", "brier_score", "tn", "fp", "fn", "tp", "total_cost",
    ]]
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    save_pr_curve(y_test, probabilities, output_dir)
    save_operating_point_plots(metrics, output_dir)
    save_calibration_plot(y_test, probabilities, output_dir)
    save_lightgbm_feature_importance(trained_models["lightgbm_standard"], output_dir)

    selected_name = "logistic_balanced"
    selected_probability = probabilities[selected_name]
    analysis = save_error_analysis(
        dataset.loc[test_indices], selected_probability, config.main_threshold, output_dir
    )
    write_manual_summary(analysis, metrics, output_dir, config.main_threshold)

    bundle = {
        "model": trained_models[selected_name],
        "model_name": selected_name,
        "model_display_name": "Balanced Logistic Regression",
        "threshold": config.main_threshold,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "config": asdict(config),
        "dataset_options": {
            column: sorted(dataset[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
    }
    joblib.dump(bundle, model_dir / "yadyar_risk_model.joblib")
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "integration_contract.json").write_text(json.dumps(prediction_contract(), indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="YadYar Lite Phase 2 pipeline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--module", default=None, help="Optional module code, e.g. DDD")
    parser.add_argument("--presentation", default=None, help="Optional presentation, e.g. 2013J")
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    config = Config(module=args.module, presentation=args.presentation, main_threshold=args.threshold)
    metrics = run_pipeline(Path(args.data_dir), Path(args.output_dir), Path(args.model_dir), config)
    print("\nPhase 2 completed.")
    print(metrics.to_string(index=False))
    print(f"\nOutputs: {Path(args.output_dir).resolve()}")
    print(f"Model: {(Path(args.model_dir) / 'yadyar_risk_model.joblib').resolve()}")


if __name__ == "__main__":
    main()
