from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .slice_analysis import save_early_assessment_slice
except ImportError:
    from slice_analysis import save_early_assessment_slice

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "outputs" / "engineered_student_dataset.csv"
MODEL_PATH = ROOT / "models" / "yadyar_risk_model.joblib"
OUTPUT_DIR = ROOT / "outputs"

BASE_CATEGORICAL = [
    "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "disability",
]
BASE_NUMERIC = [
    "num_of_prev_attempts", "studied_credits", "total_clicks_30d",
    "active_days_30d", "unique_sites_30d", "avg_clicks_per_active_day",
    "assessment_count_30d", "avg_score_30d",
]
TARGET = "at_risk"
THRESHOLD = 0.30
RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_indices(df: pd.DataFrame):
    return train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )


def binary_metrics(y_true, probability, threshold=THRESHOLD):
    pred = (probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_at_risk": precision_score(y_true, pred, zero_division=0),
        "recall_at_risk": recall_score(y_true, pred, zero_division=0),
        "f1_at_risk": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probability),
        "fn": int(((y_true.to_numpy() == 1) & (pred == 0)).sum()),
        "fp": int(((y_true.to_numpy() == 0) & (pred == 1)).sum()),
    }


def make_improved_model(categorical, numeric):
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipe, categorical),
        ("numeric", numeric_pipe, numeric),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def main():
    df = pd.read_csv(DATASET_PATH)
    train_idx, test_idx = split_indices(df)
    test = df.loc[test_idx].copy()
    y_test = test[TARGET]

    # Bonus 1: an additional slice that is different from the required engagement breakdown.
    bundle = joblib.load(MODEL_PATH)
    base_features = bundle["features"]
    base_model = bundle["model"]
    base_prob = base_model.predict_proba(test[base_features])[:, 1]
    test["probability"] = base_prob
    test["prediction"] = (base_prob >= THRESHOLD).astype(int)
    slice_df = save_early_assessment_slice(test, OUTPUT_DIR)

    # Bonus 2: lightweight feature improvement.
    # Explicitly distinguish 'no early assessment' from a numeric score of zero.
    improved = df.copy()
    improved["has_early_assessment"] = (improved["assessment_count_30d"] > 0).astype(int)
    improved_numeric = BASE_NUMERIC + ["has_early_assessment"]
    improved_features = BASE_CATEGORICAL + improved_numeric
    train = improved.loc[train_idx]
    test2 = improved.loc[test_idx]

    improved_model = make_improved_model(BASE_CATEGORICAL, improved_numeric)
    improved_model.fit(train[improved_features], train[TARGET])
    improved_prob = improved_model.predict_proba(test2[improved_features])[:, 1]

    baseline_result = binary_metrics(y_test, base_prob)
    baseline_result.update({"model": "Balanced Logistic Regression", "feature_change": "None"})
    improved_result = binary_metrics(test2[TARGET], improved_prob)
    improved_result.update({
        "model": "Balanced Logistic + has_early_assessment",
        "feature_change": "Added binary early-assessment availability feature",
    })
    cols = ["model", "feature_change", "accuracy", "precision_at_risk", "recall_at_risk", "f1_at_risk", "pr_auc", "fn", "fp"]
    improvement_df = pd.DataFrame([baseline_result, improved_result])[cols]
    improvement_df.to_csv(OUTPUT_DIR / "bonus_feature_improvement_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    metric_plot = improvement_df.set_index("model")[["precision_at_risk", "recall_at_risk", "f1_at_risk", "pr_auc"]]
    metric_plot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Bonus improvement: explicit early-assessment availability feature")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bonus_feature_improvement.png", dpi=180)
    plt.close(fig)

    summary = {
        "threshold": THRESHOLD,
        "additional_slice": slice_df.to_dict(orient="records"),
        "feature_improvement": improvement_df.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "bonus_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Bonus analysis complete.")
    print("\nAdditional assessment-status slice:\n", slice_df.to_string(index=False))
    print("\nFeature improvement comparison:\n", improvement_df.to_string(index=False))


if __name__ == "__main__":
    main()
